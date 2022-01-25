// Frontend statements
PER_STATEMENT(FrontendExternalFuncStmt)
PER_STATEMENT(FrontendExprStmt)
PER_STATEMENT(FrontendIfStmt)
PER_STATEMENT(FrontendForStmt)
PER_STATEMENT(FrontendPrintStmt)
PER_STATEMENT(FrontendWhileStmt)
PER_STATEMENT(FrontendBreakStmt)
PER_STATEMENT(FrontendContinueStmt)
PER_STATEMENT(FrontendAllocaStmt)
PER_STATEMENT(FrontendAssignStmt)
PER_STATEMENT(FrontendEvalStmt)
PER_STATEMENT(FrontendSNodeOpStmt)  // activate, deactivate, append, clear
PER_STATEMENT(FrontendAssertStmt)
PER_STATEMENT(FrontendFuncDefStmt)
PER_STATEMENT(FrontendReturnStmt)

// Middle-end statement

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
PER_STATEMENT(ElementShuffleStmt)

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

// Quantization
PER_STATEMENT(BitStructStoreStmt)
