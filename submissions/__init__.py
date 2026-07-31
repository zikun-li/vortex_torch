# Namespace for GT submissions shipped inside the vortex_torch wheel so they resolve without a
# source checkout. sab loads the flow by file path (VortexConfig.module_path, resolved against
# vortex_torch.__file__.parent.parent -> site-packages/submissions/...), so this stays importable
# AND path-loadable and the existing module_path values keep working for both checkout and wheel.
