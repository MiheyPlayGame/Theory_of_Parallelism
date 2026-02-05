# Task 1 - Array Type Selection at Build Time
## Building the Project

### Requirements
- CMake version 3.10 or higher
- C++ compiler (MinGW, GCC, Clang, or MSVC)

### Build Instructions

1. **Create a build directory:**
   ```powershell
   mkdir build
   cd build
   ```

2. **Configure the project with CMake:**
   
   To build with **double** type (default):
   ```powershell
   cmake -G "MinGW Makefiles" ..
   ```
   
   To build with **float** type:
   ```powershell
   cmake -G "MinGW Makefiles" -DUSE_FLOAT=ON ..
   ```

3. **Build the project:**
   ```powershell
   cmake --build .
   ```

4. **Run the program:**
   ```powershell
   .\main.exe
   ```

## Execution Results

### With double type:
```
sum = 0.000000
```

### With float type:
```
sum = -0.027786
```
