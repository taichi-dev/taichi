// Frontend statements
PER_STATEMENT(FrontendIfStmt)
PER_STATEMENT(FrontendForStmt)
PER_STATEMENT(FrontendPrintStmt)
PER_STATEMENT(FrontendWhileStmt)
PER_STATEMENT(FrontendBreakStmt)
PER_STATEMENT(FrontendAllocaStmt)
PER_STATEMENT(FrontendAssignStmt)
PER_STATEMENT(FrontendAtomicStmt)
PER_STATEMENT(FrontendEvalStmt)
PER_STATEMENT(FrontendSNodeOpStmt)  // activate, deactivate, append, clear
PER_STATEMENT(FrontendAssertStmt)
PER_STATEMENT(FrontendArgStoreStmt)

// Midend statement

// Without per-lane attributes
PER_STATEMENT(RangeForStmt)
PER_STATEMENT(StructForStmt)
PER_STATEMENT(IfStmt)
PER_STATEMENT(WhileStmt)
PER_STATEMENT(WhileControlStmt)

PER_STATEMENT(ArgLoadStmt)
PER_STATEMENT(ExternalPtrStmt)
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
PER_STATEMENT(AssertStmt)
PER_STATEMENT(ArgStoreStmt)

// SNode Micro Ops
PER_STATEMENT(GetRootStmt)
PER_STATEMENT(IntegerOffsetStmt)
PER_STATEMENT(OffsetAndExtractBitsStmt)
PER_STATEMENT(LinearizeStmt)
PER_STATEMENT(SNodeLookupStmt)
PER_STATEMENT(GetChStmt)

// With per-lane attributes
PER_STATEMENT(LocalLoadStmt)
PER_STATEMENT(GlobalPtrStmt)
PER_STATEMENT(ElementShuffleStmt)

// Pragma statements
PER_STATEMENT(PragmaSLPStmt)

// Clearer
PER_STATEMENT(ClearAllStmt)

// Offloaded
PER_STATEMENT(OffloadedStmt)
PER_STATEMENT(LoopIndexStmt)
PER_STATEMENT(GlobalTemporaryStmt)
