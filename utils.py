"""一些经常用到的工具函数"""

def align_up(v, unit_size=2):
    """Align the input variable with unit of sizes. The aligned data will always
    be larger than the inputs.

    Args:
        v: the variable to be aligned.
        unit_size: the block size of the aligned data.

    Return:
        aligned variable.
    """
    return (v + unit_size - 1) // unit_size * unit_size
