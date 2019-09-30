import pytest
import os

if __name__ == '__main__':
  for f in os.listdir(os.path.join(os.path.dirname(__file__), 'python')):
    if f.startswith('test') and f.endswith('.py'):
      pytest.main(['python/{}'.format(f)])
