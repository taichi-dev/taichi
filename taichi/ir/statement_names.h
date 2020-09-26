#pragma once

// This is a light-weighted header when we only need pointers("XxxStmt *").

TLANG_NAMESPACE_BEGIN

class AllocaStmt;
class WhileControlStmt;
class ContinueStmt;
class UnaryOpStmt;
class ArgLoadStmt;
class RandStmt;
class BinaryOpStmt;
class TernaryOpStmt;
class AtomicOpStmt;
class ExternalPtrStmt;
class GlobalPtrStmt;
class SNodeOpStmt;
class ExternalTensorShapeAlongAxisStmt;
class AssertStmt;
class ExternalFuncCallStmt;
class RangeAssumptionStmt;
class GlobalLoadStmt;
class GlobalStoreStmt;
class LocalLoadStmt;
class LocalStoreStmt;
class IfStmt;
class PrintStmt;
class RangeForStmt;
class StructForStmt;
class FuncBodyStmt;
class FuncCallStmt;
class KernelReturnStmt;
class WhileStmt;
class PragmaSLPStmt;
class ElementShuffleStmt;
class IntegerOffsetStmt;
class LinearizeStmt;
class BitExtractStmt;
class GetRootStmt;
class SNodeLookupStmt;
class GetChStmt;
class OffloadedStmt;
class LoopIndexStmt;
class LoopLinearIndexStmt;
class BlockCornerIndexStmt;
class BlockDimStmt;
class GlobalTemporaryStmt;
class ThreadLocalPtrStmt;
class BlockLocalPtrStmt;
class ClearListStmt;
class InternalFuncStmt;
class StackAllocaStmt;
class StackLoadTopStmt;
class StackLoadTopAdjStmt;
class StackPopStmt;
class StackPushStmt;
class StackAccAdjointStmt;

TLANG_NAMESPACE_END
