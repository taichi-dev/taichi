// Frontend statements
#include "frontend_statements.inc.h"

// Middle-end statement

// Decoration / debug statement
PER_STATEMENT(DecorationStmt)

// Without per-lane attributes
PER_STATEMENT(RangeForStmt)
PER_STATEMENT(StructForStmt)
PER_STATEMENT(MeshForStmt)
PER_STATEMENT(IfStmt)
PER_STATEMENT(WhileStmt)
PER_STATEMENT(WhileControlStmt)
PER_STATEMENT(ContinueStmt)
PER_STATEMENT(FuncCallStmt)
PER_STATEMENT(ReturnStmt)

PER_STATEMENT(ArgLoadStmt)
PER_STATEMENT(ReferenceStmt)
PER_STATEMENT(ExternalPtrStmt)
PER_STATEMENT(PtrOffsetStmt)
PER_STATEMENT(ConstStmt)
PER_STATEMENT(AllocaStmt)
PER_STATEMENT(UnaryOpStmt)
PER_STATEMENT(BinaryOpStmt)
PER_STATEMENT(TernaryOpStmt)
PER_STATEMENT(PrintStmt)
PER_STATEMENT(RandStmt)
PER_STATEMENT(GlobalLoadStmt)
PER_STATEMENT(GlobalStoreStmt)
PER_STATEMENT(AtomicOpStmt)
PER_STATEMENT(LocalStoreStmt)
PER_STATEMENT(SNodeOpStmt)
PER_STATEMENT(RangeAssumptionStmt)
PER_STATEMENT(LoopUniqueStmt)
PER_STATEMENT(AssertStmt)
PER_STATEMENT(ExternalFuncCallStmt)
PER_STATEMENT(ExternalTensorShapeAlongAxisStmt)
PER_STATEMENT(MatrixInitStmt)

// Locals with reverse-mode autodiff
PER_STATEMENT(AdStackAllocaStmt)
PER_STATEMENT(AdStackLoadTopStmt)
PER_STATEMENT(AdStackLoadTopAdjStmt)
PER_STATEMENT(AdStackPopStmt)
PER_STATEMENT(AdStackPushStmt)
PER_STATEMENT(AdStackAccAdjointStmt)

// SNode Micro Ops
PER_STATEMENT(GetRootStmt)
PER_STATEMENT(IntegerOffsetStmt)
PER_STATEMENT(BitExtractStmt)
PER_STATEMENT(LinearizeStmt)
PER_STATEMENT(SNodeLookupStmt)
PER_STATEMENT(GetChStmt)

// With per-lane attributes
PER_STATEMENT(LocalLoadStmt)
PER_STATEMENT(GlobalPtrStmt)

// Offloaded
PER_STATEMENT(OffloadedStmt)
PER_STATEMENT(MeshRelationAccessStmt)
PER_STATEMENT(MeshIndexConversionStmt)
PER_STATEMENT(MeshPatchIndexStmt)
PER_STATEMENT(LoopIndexStmt)
PER_STATEMENT(LoopLinearIndexStmt)
PER_STATEMENT(GlobalThreadIndexStmt)
PER_STATEMENT(BlockCornerIndexStmt)
PER_STATEMENT(GlobalTemporaryStmt)
PER_STATEMENT(ClearListStmt)

// Local storage
PER_STATEMENT(ThreadLocalPtrStmt)
PER_STATEMENT(BlockLocalPtrStmt)

// Special
PER_STATEMENT(InternalFuncStmt)

PER_STATEMENT(TexturePtrStmt)
PER_STATEMENT(TextureOpStmt)

// Quantization
PER_STATEMENT(BitStructStoreStmt)
