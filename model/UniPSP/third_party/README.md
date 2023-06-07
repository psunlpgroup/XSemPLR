packages(fbthrift, fizz, fmt etc) in this third_party can all be git cloned by the command:
```shell
git submodule update --init --recursive
```
But some of them are modified by Tianbao Xie for "more friendly use", since some of them must be evaluated by creating some sort of files...
these packages are "dart, e2e"

And we will make sure the config for all submodules are right in the end.