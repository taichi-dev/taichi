Sparse Computation
===============================================

.. warning::

  The Taichi compiler backend migrated from source-to-source compilation to LLVM for compilation speed and portability.
  Sparse computation with the new LLVM backend is not yet fully implemented on multithreaded CPUs and GPUs.

  If you are interested in sparse computation, please read our `paper <http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf>`_, watch the `introduction video <https://www.youtube.com/watch?v=wKw8LMF3Djo>`_, or check out
  the SIGGRAPH Asia 2019 `slides <http://taichi.graphics/wp-content/uploads/2019/12/taichi_slides.pdf>`_.

  The legacy source-to-source backend (commit ``dc162e11``) provides full sparse computation functionality. However, since little engineering has been made to make that commit portable,
  we suggest waiting until the LLVM version of sparse computation is fully implemented.

  Sparse computation functionalities with the new LLVM backend will be back online by the end of December 2019.
