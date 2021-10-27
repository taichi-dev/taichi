curl https://sdk.lunarg.com/sdk/download/1.2.189.0/mac/vulkansdk-macos-1.2.189.0.dmg -o vk.dmg
sudo hdiutil attach vk.dmg
cp -r  /Volumes/vulkansdk-macos-1.2.189.0/InstallVulkan.app .
sudo ./InstallVulkan.app/Contents/MacOS/InstallVulkan --root ~/VulkanSDK/1.2.189.0 --accept-licenses --default-answer --confirm-command install

printf '%s' '
VULKAN_SDK=~/VulkanSDK/1.2.189.0/macOS
export VULKAN_SDK
PATH="$PATH:$VULKAN_SDK/bin"
export PATH
DYLD_LIBRARY_PATH="$VULKAN_SDK/lib:${DYLD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH
VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
export VK_LAYER_PATH
VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export VK_ICD_FILENAMES
' >> ~/.zshrc
