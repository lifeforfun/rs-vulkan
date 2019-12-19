#!/bin/bash
export VULKAN_SDK=$HOME/workspace/test/sdk-vulkan/macOS
export PATH=$VULKAN_SDK/bin:$PATH
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib:$DYLD_LIBRARY_PATH
export VK_ICD_FILENAMES=$VULKAN_SDK/etc/vulkan/icd.d/MoltenVK_icd.json
export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH