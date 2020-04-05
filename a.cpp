#include <iostream>
#include <memory>

struct Bar
{
  int x;
  Bar(int x) : x(x) {}
};

struct Foo
{
  std::unique_ptr<Bar> bar;
  Foo(int x) : bar(std::make_unique<Bar>(x)) {}
  std::unique_ptr<Bar> get_bar() {
    return std::move(bar);
  }
};

int main()
{
  Foo foo(233);
  auto bar = foo.get_bar();
  std::cout << bar->x << std::endl;
}
