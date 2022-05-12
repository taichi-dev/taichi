#pragma once

#ifdef _WIN64
#define VK_USE_PLATFORM_WIN32_KHR 1
#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <VersionHelpers.h>
#endif

#include "taichi/backends/vulkan/vulkan_common.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace taichi {
namespace lang {
namespace vulkan {

class VulkanEnvSettings {
 public:
  // This version number is used to create a vkInstance, it should be
  // the highest API version that is designed to use.
  // Reference:
  // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkApplicationInfo.html
  static constexpr uint32_t kApiVersion() {
    return VK_API_VERSION_1_3;
  }
};

#ifdef _WIN64

// this class is needed for Vulkan-Cuda interop on Windows
// the implementation is taken from
// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/vulkanImageCUDA
class WindowsSecurityAttributes {
 protected:
  SECURITY_ATTRIBUTES security_attributes;
  PSECURITY_DESCRIPTOR security_descriptor;

 public:
  WindowsSecurityAttributes();
  SECURITY_ATTRIBUTES *operator&();
  ~WindowsSecurityAttributes();
};

inline WindowsSecurityAttributes::WindowsSecurityAttributes() {
  security_descriptor = (PSECURITY_DESCRIPTOR)calloc(
      1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));

  PSID *sid =
      (PSID *)((PBYTE)security_descriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
  PACL *acl = (PACL *)((PBYTE)sid + sizeof(PSID *));

  InitializeSecurityDescriptor(security_descriptor,
                               SECURITY_DESCRIPTOR_REVISION);

  SID_IDENTIFIER_AUTHORITY sid_identifier_auth = SECURITY_WORLD_SID_AUTHORITY;
  AllocateAndInitializeSid(&sid_identifier_auth, 1, SECURITY_WORLD_RID, 0, 0, 0,
                           0, 0, 0, 0, sid);

  EXPLICIT_ACCESS explicit_access;
  ZeroMemory(&explicit_access, sizeof(EXPLICIT_ACCESS));
  explicit_access.grfAccessPermissions =
      STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
  explicit_access.grfAccessMode = SET_ACCESS;
  explicit_access.grfInheritance = INHERIT_ONLY;
  explicit_access.Trustee.TrusteeForm = TRUSTEE_IS_SID;
  explicit_access.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
  explicit_access.Trustee.ptstrName = (LPTSTR)*sid;

  SetEntriesInAcl(1, &explicit_access, NULL, acl);

  SetSecurityDescriptorDacl(security_descriptor, TRUE, *acl, FALSE);

  security_attributes.nLength = sizeof(security_attributes);
  security_attributes.lpSecurityDescriptor = security_descriptor;
  security_attributes.bInheritHandle = TRUE;
}

inline SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() {
  return &security_attributes;
}

inline WindowsSecurityAttributes::~WindowsSecurityAttributes() {
  PSID *sid =
      (PSID *)((PBYTE)security_descriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
  PACL *acl = (PACL *)((PBYTE)sid + sizeof(PSID *));

  if (*sid) {
    FreeSid(*sid);
  }
  if (*acl) {
    LocalFree(*acl);
  }
  free(security_descriptor);
}

#endif  //_WIN64

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
