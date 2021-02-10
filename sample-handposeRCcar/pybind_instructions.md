## Prerequisites
- install pybind11 on board

## Compile uart.cpp with pybind instructions
- go to `./sample-handposeRC/Atlas200DK/sample-handposeRC/src/` 
- run following command to compile file into python module
```
c++ -O3 -Wall -shared -std=c++11 -fPIC -I/usr/include/python3.6m -I/home/HwHiAiUser/.local/lib/python3.6/site-packages/pybind11/include -I ../inc-python  uart-python.cpp -o uart.cpython-36m-aarch64-linux-gnu.so
```

- then in the same directory as the .so file, just `import uart`

### Example import
```
import uart

# create uart object
uart_conn = uart.UART()

# run uart command
uart_conn.uart_open()

```

- see uart-python.cpp for function definitions



