## Maps

C# methods for PySymGym training.

## Structure

- `dataset.json` — training episodes configuration
- `Root/bin/Release/net8.0/` — compiled DLL
- `src/` — source code (submodules)

## Dataset Structure
`dataset.json` contains training episodes with the following fields:

``` json:
- StepsToPlay: total steps
- StepsToStart: baseline steps
- AssemblyFullName: DLL name
- NameOfObjectToCover: namespace.class.method
- DefaultSearcher: BFS/DFS
- MapName: MethodName_Steps_Strategy (or MethodName_0)
```
## Root Project

`Root/Root.csproj` is a .NET project that references all submodule libraries. When you build it, all dependencies are compiled into `Root/bin/Release/net8.0/`.

Build all libraries:
```bash
cd Root
dotnet build -c Release
```
After adding or updating submodules, rebuild the project to refresh the compiled assemblies.

## Method Verification

Before adding new methods to the dataset, they must be verified to ensure they execute correctly in the V# symbolic execution engine and produce non‑zero coverage.

Create a `methods.csv` file listing the methods to test:
```csv
dll,method
Library.dll,namespace.class.method
```
Run the verification with runstrat:
```bash
python3 runstrat.py \
    -s ExecutionTreeContributedCoverage \   # search strategy
    -t 120 \                                 # timeout (seconds)
    -ps ../../ \                              # path to PySymGym root
    -as ../../maps/DotNet/Maps/Root/bin/Release/net8.0 \  # path to assemblies
    methods.csv                               # file with methods
```
## Submodules
 
- https://github.com/JetBrains/rd.git
  
- https://github.com/aalhour/C-Sharp-Algorithms.git

- https://github.com/justcoding121/advanced-algorithms.git
  
- https://github.com/TASEmulators/BizHawk.git
  
- https://github.com/mxprshn/StandaloneUnity.git
  
- https://github.com/mbdavid/LiteDB.git
  
- https://github.com/SaeedGz98/stralgo.git
  
- https://github.com/cat923/C-Sharp.git
