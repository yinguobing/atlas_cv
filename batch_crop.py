"""Minimal code showing how to crop multiple areas from one image with pyACL."""
import acl
import cv2
import numpy as np

import atlas.common.atlas_utils.constants as constants
import atlas.common.atlas_utils.utils as utils


def create_buffer(width, height, align_w=16, align_h=2):
    """Create an alinged buffer for image.

    Buffer size will be width_stride * height_stride, which may be larger than
    width * height.

    Args:
        width: image width.
        height: image height.
        align_w: the width unit for alignement.
        align_h: the height unit for alignement.

    Returns:
        ptr_on_device: pointer on the device.
        buf_size: the buffer size.
        width_stride: the width of the image after alignment.
        height_stride: the height of the image after alignment.
    """
    # Calculate buffer size.
    width_stride = utils.align_up(width, align_w)
    height_stride = utils.align_up(height, align_h)
    buf_size = (width_stride * height_stride * 3) // 2

    # Allocate memory
    ptr_on_device, ret = acl.media.dvpp_malloc(buf_size)
    utils.check_ret("acl.media.dvpp_malloc", ret)

    return ptr_on_device, buf_size, width_stride, height_stride


def setup_desc(desc, buf, buf_size, width, height, width_stride, height_stride):
    """Setup the image description object.

    The image SHOULD be in YUV420SP format.

    Args:
        desc: the image description.
        buf: image pointer.
        width: the width of the image.
        height: the height of the image.
        width_stride: the width_stride of the image.
        height_stride: the height_stride of the image.
    """
    acl.media.dvpp_set_pic_desc_data(desc, buf)
    acl.media.dvpp_set_pic_desc_size(desc, buf_size)
    acl.media.dvpp_set_pic_desc_format(
        desc, constants.PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    acl.media.dvpp_set_pic_desc_width(desc, width)
    acl.media.dvpp_set_pic_desc_height(desc, height)
    acl.media.dvpp_set_pic_desc_width_stride(desc, width_stride)
    acl.media.dvpp_set_pic_desc_height_stride(desc, height_stride)


def batch_crop(yuv, width, height, boxes, stream, dvpp_channel_desc):

    # 准备待裁切图像

    # 虽然你已经将YUV图像读入内存，但是硬件加速的裁切操作需要在Ascend芯片上进行，这意味着你需
    # 要把图像数据从主机(host)一侧拷贝到Ascend芯片(Device)一侧。另外，Device一侧的数据并非
    # Numpy数组，在这里生存的图像有自己独特的“描述方式”。
    box_count = len(boxes)
    image_count = 1

    roi_num_list = []
    corp_area_list = []

    # 创建图片批处理输入输出描述
    inputs_desc = acl.media.dvpp_create_batch_pic_desc(image_count)
    outputs_desc = acl.media.dvpp_create_batch_pic_desc(box_count)

    # 逐个准备待处理的输入图像
    buffers_in = []
    for i in range(image_count):

        # 获取单个图片描述对象
        input_desc = acl.media.dvpp_get_pic_desc(inputs_desc, i)
        assert input_desc is not None

        # 为输入图像申请内存并设定描述对象
        h, w, = height, width
        ptr_in, buf_size, stride_w, stride_h = create_buffer(w, h, 16, 2)
        setup_desc(input_desc, ptr_in, buf_size, w, h, stride_w, stride_h)

        # 拷贝数据到Device并记录该数据位置
        buffer_size_in = yuv.itemsize * yuv.size
        ptr_yuv = acl.util.numpy_to_ptr(yuv)
        ret = acl.rt.memcpy(ptr_in, buffer_size_in, ptr_yuv, buffer_size_in,
                            constants.ACL_MEMCPY_HOST_TO_DEVICE)
        assert ret == 0, "图像内存拷贝失败。"
        buffers_in.append(ptr_in)

        # 记录需要裁切的区域数量
        roi_num_list.append(box_count)

    # 逐个设置输出图片描述
    buffers_out = []
    for i in range(box_count):

        # 获取输出图像描述对象
        output_desc = acl.media.dvpp_get_pic_desc(outputs_desc, i)
        assert output_desc is not None

        # 自定义方法申请内存并设置输出图片描述
        x1, x2, y1, y2 = map(utils.align_up, boxes[i])
        w = x2 - x1
        h = y2 - y1
        ptr_out, buf_size, stride_w, stride_h = create_buffer(w, h, 16, 2)
        setup_desc(output_desc, ptr_out, buf_size, w, h, stride_w, stride_h)
        buffers_out.append(ptr_out)

        # 设定裁切区域
        crop_area = acl.media.dvpp_create_roi_config(x1, x2-1, y1, y2-1)
        corp_area_list.append(crop_area)

    # 执行异步裁切
    outputs_desc, ret = acl.media.dvpp_vpc_batch_crop_async(
        dvpp_channel_desc,
        inputs_desc,
        roi_num_list,
        outputs_desc,
        corp_area_list,
        stream)
    utils.check_ret("acl.media.dvpp_vpc_batch_crop_async", ret)

    # 同步裁切任务结果
    ret = acl.rt.synchronize_stream(stream)
    utils.check_ret("acl.rt.synchronize_stream", ret)

    # 查阅结果。

    # 获取裁切后的图像信息。这一步需要将YUV420SP转换为BGR格式。
    outputs = []
    for i in range(box_count):
        
        # 获得裁切图像描述对象
        output_desc = acl.media.dvpp_get_pic_desc(outputs_desc, i)
        assert output_desc is not None
        ret_code = acl.media.dvpp_get_pic_desc_ret_code(output_desc)
        utils.check_ret("acl.media.dvpp_get_pic_desc_ret_code", ret_code)

        # 获取裁切后的图像信息
        data = acl.media.dvpp_get_pic_desc_data(output_desc)
        data_size = acl.media.dvpp_get_pic_desc_size(output_desc)
        width_stride = acl.media.dvpp_get_pic_desc_width_stride(output_desc)
        height_stride = acl.media.dvpp_get_pic_desc_height_stride(output_desc)

        # 拷贝数据到主机一侧并转换为Numpy数组。
        np_pic = np.zeros(data_size, dtype=np.byte)
        np_pic_ptr = acl.util.numpy_to_ptr(np_pic)
        ret = acl.rt.memcpy(np_pic_ptr, data_size,
                            data, data_size,
                            constants.ACL_MEMCPY_DEVICE_TO_HOST)
        utils.check_ret("acl.rt.memcpy", ret)

        result_yuv = np.reshape(
            np_pic, (int((height_stride)*3/2), int(width_stride))).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_yuv, cv2.COLOR_YUV2BGR_NV12)
        org_w = acl.media.dvpp_get_pic_desc_width(output_desc)
        org_h = acl.media.dvpp_get_pic_desc_height(output_desc)
        result_bgr = result_bgr[0:org_h, 0:org_w, :]
        outputs.append(result_bgr)

    # 收尾、释放资源

    # 释放批量图片描述
    acl.media.dvpp_destroy_batch_pic_desc(inputs_desc)
    acl.media.dvpp_destroy_batch_pic_desc(outputs_desc)

    # 释放ROI配置
    for i in range(len(corp_area_list)):
        ret = acl.media.dvpp_destroy_roi_config(corp_area_list[i])
        assert ret == 0, "释放ROI错误。"

    # 释放Device侧内存
    for p in buffers_in:
        ret = acl.media.dvpp_free(p)
        assert ret == 0, "释放输入内存错误。"

    for p in buffers_out:
        ret = acl.media.dvpp_free(p)
        assert ret == 0, "释放输出内存错误。"

    return outputs


def main():
    # 第一步：初始化

    # Ascend设备上运行至少需要初始化4个对象：ACL、Device、Context和Stream。这个过
    # 程对于Python开发者可能比较陌生。好在这个过程是一次性的。

    # ACL初始化。ACL是操作Ascend设备的Python库，有点像OpenCV。
    ret = acl.init()
    assert ret == 0, "初始化ACL失败。"

    # Device初始化。一张Atlas加速卡上可能有多个Ascend芯片可以使用。在这里指定你要选用的芯片
    # 序号。
    device_id = 0
    ret = acl.rt.set_device(device_id)
    assert ret == 0, "初始化Device失败。"

    # Context初始化。Context可以看做是程序运行的小环境，它起着隔离作用。
    context, ret = acl.rt.create_context(device_id)
    assert ret == 0, "初始化Context失败。"

    # Stream初始化。Stream多用作异步操作的同步。
    stream, ret = acl.rt.create_stream()
    assert ret == 0, "初始化Stream失败。"

    # 创建图片数据处理通道时的通道描述信息。
    dvpp_channel_desc = acl.media.dvpp_create_channel_desc()

    # 创建图片数据处理的通道。
    ret = acl.media.dvpp_create_channel(dvpp_channel_desc)
    utils.check_ret("acl.media.dvpp_create_channel", ret)

    # 成功执行到这一步，初始化完成。

    # 第二步：图像读入

    # ACL中专门为图像相关操作提供了硬件加速模块。这些加速有一个前提条件：图像需要为YUV格式。
    # 另外请注意，OpenCV暂时无法将BGR图像转换为ACL支持的YUV420SP格式。所以我们将直接读入一
    # 张YUV图像。

    # 读入一张YUV图像。
    img_file = "wood_rabbit_1024_1068_nv12.yuv"
    yuv = np.fromfile(img_file, dtype=np.byte)

    # 为了验证读取结果，将读入的YUV图像转换为你熟悉的OpenCV图像：NumpyArray。并在窗口中查看
    # 图像内容。
    yuv = np.reshape(yuv, (int((1072)*3/2), int(1024))).astype(np.uint8)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

    # 第三步：图像裁切

    # 指定批量裁切区域。
    boxes = [[421, 421+348, 441, 441+259],
             [359, 359+255, 369, 369+257], 
             [462, 462+255, 521, 523+257],]
    height, width, _ = bgr.shape

    # 开始裁切
    outputs = batch_crop(yuv, width, height, boxes, stream, dvpp_channel_desc)

    # 查看裁切结果
    for i, img in enumerate(outputs):
        cv2.imshow('Crop_{}'.format(i), img)
    cv2.imshow('Original', bgr)
    cv2.waitKey()

    # 第四步：收尾、释放资源

    # 释放操作通道
    if dvpp_channel_desc:
        ret = acl.media.dvpp_destroy_channel(dvpp_channel_desc)
        assert ret == 0
        ret = acl.media.dvpp_destroy_channel_desc(dvpp_channel_desc)
        assert ret == 0

    # 解码结束后，释放资源，包括输入/输出图片的描述信息、输入/输出内存、通道描述信息、通道等
    # 释放运行管理资源
    ret = acl.rt.destroy_stream(stream)
    ret = acl.rt.destroy_context(context)
    ret = acl.rt.reset_device(device_id)
    ret = acl.finalize()

    print('结束。')


if __name__ == "__main__":
    main()
