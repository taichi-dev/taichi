Some notes about the current implementation

There are lots of `from taichi._kernels import xx`. Unfortunately, this cannot be moved into the top of the file, and has to be delayed inside the function. Otherwise, it would result in some cyclic import issue that is not trivially resolvable.
