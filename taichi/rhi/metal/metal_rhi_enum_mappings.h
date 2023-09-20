
MTLPrimitiveType primtype2mtl(TopologyType type) {
  static const std::unordered_map<TopologyType, MTLPrimitiveType> map = {
      {TopologyType::Triangles, MTLPrimitiveType::MTLPrimitiveTypeTriangle},
      {TopologyType::Lines, MTLPrimitiveType::MTLPrimitiveTypeLine},
      {TopologyType::Points, MTLPrimitiveType::MTLPrimitiveTypePoint},
  };
  auto it = map.find(type);
  RHI_ASSERT(it != map.end());
  return it->second;
}

MTLPrimitiveTopologyClass topotype2mtl(TopologyType type) {
  static const std::unordered_map<TopologyType, MTLPrimitiveTopologyClass> map =
      {
          {TopologyType::Triangles,
           MTLPrimitiveTopologyClass::MTLPrimitiveTopologyClassTriangle},
          {TopologyType::Lines,
           MTLPrimitiveTopologyClass::MTLPrimitiveTopologyClassLine},
          {TopologyType::Points,
           MTLPrimitiveTopologyClass::MTLPrimitiveTopologyClassPoint},
      };
  auto it = map.find(type);
  RHI_ASSERT(it != map.end());
  return it->second;
}

MTLBlendOperation blendop2mtl(BlendOp op) {
  static const std::unordered_map<BlendOp, MTLBlendOperation> map = {
      {BlendOp::add, MTLBlendOperation::MTLBlendOperationAdd},
      {BlendOp::subtract, MTLBlendOperation::MTLBlendOperationSubtract},
      {BlendOp::reverse_subtract,
       MTLBlendOperation::MTLBlendOperationReverseSubtract},
      {BlendOp::min, MTLBlendOperation::MTLBlendOperationMin},
      {BlendOp::max, MTLBlendOperation::MTLBlendOperationMax},
  };
  auto it = map.find(op);
  RHI_ASSERT(it != map.end());
  return it->second;
}

MTLBlendFactor blendfactor2mtl(BlendFactor factor) {
  static const std::unordered_map<BlendFactor, MTLBlendFactor> map = {
      {BlendFactor::zero, MTLBlendFactor::MTLBlendFactorZero},
      {BlendFactor::one, MTLBlendFactor::MTLBlendFactorOne},
      {BlendFactor::src_color, MTLBlendFactor::MTLBlendFactorSourceColor},
      {BlendFactor::one_minus_src_color,
       MTLBlendFactor::MTLBlendFactorOneMinusSourceColor},
      {BlendFactor::dst_color, MTLBlendFactor::MTLBlendFactorDestinationColor},
      {BlendFactor::one_minus_dst_color,
       MTLBlendFactor::MTLBlendFactorOneMinusDestinationColor},
      {BlendFactor::src_alpha, MTLBlendFactor::MTLBlendFactorSourceAlpha},
      {BlendFactor::one_minus_src_alpha,
       MTLBlendFactor::MTLBlendFactorOneMinusSourceAlpha},
      {BlendFactor::dst_alpha, MTLBlendFactor::MTLBlendFactorDestinationAlpha},
      {BlendFactor::one_minus_dst_alpha,
       MTLBlendFactor::MTLBlendFactorOneMinusDestinationAlpha},
  };
  auto it = map.find(factor);
  RHI_ASSERT(it != map.end());
  return it->second;
}

MTLTriangleFillMode fillmode2mtl(PolygonMode mode) {
  static const std::unordered_map<PolygonMode, MTLTriangleFillMode> map = {
      {PolygonMode::Fill, MTLTriangleFillMode::MTLTriangleFillModeFill},
      {PolygonMode::Line, MTLTriangleFillMode::MTLTriangleFillModeLines},
      {PolygonMode::Point, MTLTriangleFillMode::MTLTriangleFillModeLines},
  };
  auto it = map.find(mode);
  RHI_ASSERT(it != map.end());
  return it->second;
}

MTLVertexFormat vertexformat2mtl(BufferFormat format) {
  static const std::map<BufferFormat, MTLVertexFormat> map{
      {BufferFormat::unknown, MTLVertexFormatInvalid},
      {BufferFormat::r8, MTLVertexFormatUCharNormalized},
      {BufferFormat::rg8, MTLVertexFormatUChar2Normalized},
      {BufferFormat::rgba8, MTLVertexFormatUChar4Normalized},
      {BufferFormat::rgba8srgb, MTLVertexFormatInvalid},
      {BufferFormat::bgra8, MTLVertexFormatUChar4Normalized_BGRA},
      {BufferFormat::bgra8srgb, MTLVertexFormatInvalid},
      {BufferFormat::r8u, MTLVertexFormatInvalid},
      {BufferFormat::rg8u, MTLVertexFormatInvalid},
      {BufferFormat::rgba8u, MTLVertexFormatInvalid},
      {BufferFormat::r8i, MTLVertexFormatInvalid},
      {BufferFormat::rg8i, MTLVertexFormatInvalid},
      {BufferFormat::rgba8i, MTLVertexFormatInvalid},
      {BufferFormat::r16, MTLVertexFormatUShortNormalized},
      {BufferFormat::rg16, MTLVertexFormatUShort2Normalized},
      {BufferFormat::rgb16, MTLVertexFormatUShort3Normalized},
      {BufferFormat::rgba16, MTLVertexFormatUShort4Normalized},
      {BufferFormat::r16u, MTLVertexFormatInvalid},
      {BufferFormat::rg16u, MTLVertexFormatInvalid},
      {BufferFormat::rgb16u, MTLVertexFormatInvalid},
      {BufferFormat::rgba16u, MTLVertexFormatInvalid},
      {BufferFormat::r16i, MTLVertexFormatInvalid},
      {BufferFormat::rg16i, MTLVertexFormatInvalid},
      {BufferFormat::rgb16i, MTLVertexFormatInvalid},
      {BufferFormat::rgba16i, MTLVertexFormatInvalid},
      {BufferFormat::r16f, MTLVertexFormatHalf},
      {BufferFormat::rg16f, MTLVertexFormatHalf2},
      {BufferFormat::rgb16f, MTLVertexFormatHalf3},
      {BufferFormat::rgba16f, MTLVertexFormatHalf4},
      {BufferFormat::r32u, MTLVertexFormatUInt},
      {BufferFormat::rg32u, MTLVertexFormatUInt2},
      {BufferFormat::rgb32u, MTLVertexFormatUInt3},
      {BufferFormat::rgba32u, MTLVertexFormatUInt4},
      {BufferFormat::r32i, MTLVertexFormatInt},
      {BufferFormat::rg32i, MTLVertexFormatInt2},
      {BufferFormat::rgb32i, MTLVertexFormatInt3},
      {BufferFormat::rgba32i, MTLVertexFormatInt4},
      {BufferFormat::r32f, MTLVertexFormatFloat},
      {BufferFormat::rg32f, MTLVertexFormatFloat2},
      {BufferFormat::rgb32f, MTLVertexFormatFloat3},
      {BufferFormat::rgba32f, MTLVertexFormatFloat4},
      {BufferFormat::depth16, MTLVertexFormatInvalid},
      {BufferFormat::depth24stencil8, MTLVertexFormatInvalid},
      {BufferFormat::depth32f, MTLVertexFormatInvalid},
  };
  auto it = map.find(format);
  RHI_ASSERT(it != map.end());
  return it->second;
}
MTLPixelFormat format2mtl(BufferFormat format) {
  static const std::map<BufferFormat, MTLPixelFormat> map{
      {BufferFormat::unknown, MTLPixelFormatInvalid},
      {BufferFormat::r8, MTLPixelFormatR8Unorm},
      {BufferFormat::rg8, MTLPixelFormatRG8Unorm},
      {BufferFormat::rgba8, MTLPixelFormatRGBA8Unorm},
      {BufferFormat::rgba8srgb, MTLPixelFormatRGBA8Unorm_sRGB},
      {BufferFormat::bgra8, MTLPixelFormatBGRA8Unorm},
      {BufferFormat::bgra8srgb, MTLPixelFormatBGRA8Unorm_sRGB},
      {BufferFormat::r8u, MTLPixelFormatR8Uint},
      {BufferFormat::rg8u, MTLPixelFormatRG8Uint},
      {BufferFormat::rgba8u, MTLPixelFormatRGBA8Uint},
      {BufferFormat::r8i, MTLPixelFormatR8Sint},
      {BufferFormat::rg8i, MTLPixelFormatRG8Sint},
      {BufferFormat::rgba8i, MTLPixelFormatRGBA8Sint},
      {BufferFormat::r16, MTLPixelFormatR16Unorm},
      {BufferFormat::rg16, MTLPixelFormatRG16Unorm},
      {BufferFormat::rgb16, MTLPixelFormatInvalid},
      {BufferFormat::rgba16, MTLPixelFormatRGBA16Unorm},
      {BufferFormat::r16u, MTLPixelFormatR16Uint},
      {BufferFormat::rg16u, MTLPixelFormatRG16Uint},
      {BufferFormat::rgb16u, MTLPixelFormatInvalid},
      {BufferFormat::rgba16u, MTLPixelFormatRGBA16Uint},
      {BufferFormat::r16i, MTLPixelFormatR16Sint},
      {BufferFormat::rg16i, MTLPixelFormatRG16Sint},
      {BufferFormat::rgb16i, MTLPixelFormatInvalid},
      {BufferFormat::rgba16i, MTLPixelFormatRGBA16Sint},
      {BufferFormat::r16f, MTLPixelFormatR16Float},
      {BufferFormat::rg16f, MTLPixelFormatRG16Float},
      {BufferFormat::rgb16f, MTLPixelFormatInvalid},
      {BufferFormat::rgba16f, MTLPixelFormatRGBA16Float},
      {BufferFormat::r32u, MTLPixelFormatR32Uint},
      {BufferFormat::rg32u, MTLPixelFormatRG32Uint},
      {BufferFormat::rgb32u, MTLPixelFormatInvalid},
      {BufferFormat::rgba32u, MTLPixelFormatRGBA32Uint},
      {BufferFormat::r32i, MTLPixelFormatR32Sint},
      {BufferFormat::rg32i, MTLPixelFormatRG32Sint},
      {BufferFormat::rgb32i, MTLPixelFormatInvalid},
      {BufferFormat::rgba32i, MTLPixelFormatRGBA32Sint},
      {BufferFormat::r32f, MTLPixelFormatR32Float},
      {BufferFormat::rg32f, MTLPixelFormatRG32Float},
      {BufferFormat::rgb32f, MTLPixelFormatInvalid},
      {BufferFormat::rgba32f, MTLPixelFormatRGBA32Float},
      {BufferFormat::depth16, MTLPixelFormatDepth16Unorm},
      {BufferFormat::depth24stencil8, MTLPixelFormatInvalid},
      {BufferFormat::depth32f, MTLPixelFormatDepth32Float},
  };
  auto it = map.find(format);
  RHI_ASSERT(it != map.end());
  return it->second;
}
BufferFormat mtl2format(MTLPixelFormat format) {
  static const std::map<BufferFormat, MTLPixelFormat> map{
      {BufferFormat::unknown, MTLPixelFormatInvalid},
      {BufferFormat::r8, MTLPixelFormatR8Unorm},
      {BufferFormat::rg8, MTLPixelFormatRG8Unorm},
      {BufferFormat::rgba8, MTLPixelFormatRGBA8Unorm},
      {BufferFormat::rgba8srgb, MTLPixelFormatRGBA8Unorm_sRGB},
      {BufferFormat::bgra8, MTLPixelFormatBGRA8Unorm},
      {BufferFormat::bgra8srgb, MTLPixelFormatBGRA8Unorm_sRGB},
      {BufferFormat::r8u, MTLPixelFormatR8Uint},
      {BufferFormat::rg8u, MTLPixelFormatRG8Uint},
      {BufferFormat::rgba8u, MTLPixelFormatRGBA8Uint},
      {BufferFormat::r8i, MTLPixelFormatR8Sint},
      {BufferFormat::rg8i, MTLPixelFormatRG8Sint},
      {BufferFormat::rgba8i, MTLPixelFormatRGBA8Sint},
      {BufferFormat::r16, MTLPixelFormatR16Unorm},
      {BufferFormat::rg16, MTLPixelFormatRG16Unorm},
      {BufferFormat::rgb16, MTLPixelFormatInvalid},
      {BufferFormat::rgba16, MTLPixelFormatRGBA16Unorm},
      {BufferFormat::r16u, MTLPixelFormatR16Uint},
      {BufferFormat::rg16u, MTLPixelFormatRG16Uint},
      {BufferFormat::rgb16u, MTLPixelFormatInvalid},
      {BufferFormat::rgba16u, MTLPixelFormatRGBA16Uint},
      {BufferFormat::r16i, MTLPixelFormatR16Sint},
      {BufferFormat::rg16i, MTLPixelFormatRG16Sint},
      {BufferFormat::rgb16i, MTLPixelFormatInvalid},
      {BufferFormat::rgba16i, MTLPixelFormatRGBA16Sint},
      {BufferFormat::r16f, MTLPixelFormatR16Float},
      {BufferFormat::rg16f, MTLPixelFormatRG16Float},
      {BufferFormat::rgb16f, MTLPixelFormatInvalid},
      {BufferFormat::rgba16f, MTLPixelFormatRGBA16Float},
      {BufferFormat::r32u, MTLPixelFormatR32Uint},
      {BufferFormat::rg32u, MTLPixelFormatRG32Uint},
      {BufferFormat::rgb32u, MTLPixelFormatInvalid},
      {BufferFormat::rgba32u, MTLPixelFormatRGBA32Uint},
      {BufferFormat::r32i, MTLPixelFormatR32Sint},
      {BufferFormat::rg32i, MTLPixelFormatRG32Sint},
      {BufferFormat::rgb32i, MTLPixelFormatInvalid},
      {BufferFormat::rgba32i, MTLPixelFormatRGBA32Sint},
      {BufferFormat::r32f, MTLPixelFormatR32Float},
      {BufferFormat::rg32f, MTLPixelFormatRG32Float},
      {BufferFormat::rgb32f, MTLPixelFormatInvalid},
      {BufferFormat::rgba32f, MTLPixelFormatRGBA32Float},
      {BufferFormat::depth16, MTLPixelFormatDepth16Unorm},
      {BufferFormat::depth24stencil8, MTLPixelFormatInvalid},
      {BufferFormat::depth32f, MTLPixelFormatDepth32Float},
  };
  auto it = std::find_if(map.begin(), map.end(),
                         [&format](auto &&p) { return p.second == format; });
  RHI_ASSERT(it != map.end());
  return it->first;
}
MTLTextureType dimension2mtl(ImageDimension dimension) {
  static const std::map<ImageDimension, MTLTextureType> map = {
      {ImageDimension::d1D, MTLTextureType1D},
      {ImageDimension::d2D, MTLTextureType2D},
      {ImageDimension::d3D, MTLTextureType3D},
  };
  auto it = map.find(dimension);
  RHI_ASSERT(it != map.end());
  return it->second;
}
MTLTextureUsage usage2mtl(ImageAllocUsage usage) {
  MTLTextureUsage out = 0;
  if (usage & ImageAllocUsage::Sampled) {
    out |= MTLTextureUsageShaderRead;
  }
  if (usage & ImageAllocUsage::Storage) {
    out |= MTLTextureUsageShaderWrite;
  }
  if (usage & ImageAllocUsage::Attachment) {
    out |= MTLTextureUsageRenderTarget;
  }
  return out;
}

uint32_t mtlformat2size(MTLPixelFormat format) {
  static const std::map<MTLPixelFormat, uint32_t> map{
      {MTLPixelFormatInvalid, 0},         {MTLPixelFormatR8Unorm, 1},
      {MTLPixelFormatRG8Unorm, 2},        {MTLPixelFormatRGBA8Unorm, 4},
      {MTLPixelFormatRGBA8Unorm_sRGB, 4}, {MTLPixelFormatBGRA8Unorm, 4},
      {MTLPixelFormatBGRA8Unorm_sRGB, 4}, {MTLPixelFormatR8Uint, 1},
      {MTLPixelFormatRG8Uint, 2},         {MTLPixelFormatRGBA8Uint, 4},
      {MTLPixelFormatR8Sint, 1},          {MTLPixelFormatRG8Sint, 2},
      {MTLPixelFormatRGBA8Sint, 4},       {MTLPixelFormatR16Unorm, 2},
      {MTLPixelFormatRG16Unorm, 4},       {MTLPixelFormatRGBA16Unorm, 8},
      {MTLPixelFormatR16Uint, 2},         {MTLPixelFormatRG16Uint, 4},
      {MTLPixelFormatRGBA16Uint, 8},      {MTLPixelFormatR16Sint, 2},
      {MTLPixelFormatRG16Sint, 4},        {MTLPixelFormatRGBA16Sint, 8},
      {MTLPixelFormatR16Float, 2},        {MTLPixelFormatRG16Float, 4},
      {MTLPixelFormatRGBA16Float, 8},     {MTLPixelFormatR32Uint, 4},
      {MTLPixelFormatRG32Uint, 8},        {MTLPixelFormatRGBA32Uint, 16},
      {MTLPixelFormatR32Sint, 4},         {MTLPixelFormatRG32Sint, 8},
      {MTLPixelFormatRGBA32Sint, 16},     {MTLPixelFormatR32Float, 4},
      {MTLPixelFormatRG32Float, 8},       {MTLPixelFormatRGBA32Float, 16},
      {MTLPixelFormatDepth16Unorm, 2},    {MTLPixelFormatDepth32Float, 4},
  };
  auto it = map.find(format);
  RHI_ASSERT(it != map.end());
  return it->second;
}
