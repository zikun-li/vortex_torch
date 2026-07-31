# Package marker so setuptools packages.find includes this GT submission in the wheel.
# ground_truth_kernel_topk.py is loaded by path (VortexConfig.module_path); this __init__ only
# makes the directory a discoverable package for packaging.
