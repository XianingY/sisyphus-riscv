

【提交要求】

总体要求：实现一个编译器，将SysY2026语言（初赛沿用SysY2022）的代码翻译为RISC-V汇编代码。仅可使用C、C++、Java或Rust。

提交的编译器应支持指定的SysY2026语言（初赛沿用SysY2022）。关于SysY2022语言的定义请参考下方附件中提供的详细说明文件"SysY2022语言定义"。

提交的编译器需要具有词法分析、语法分析、语义分析、目标代码生成与优化等功能。

RISC-V硬件赛道，对于正确编译通过的SysY2022基准测试程序，应生成符合要求的RISC-V汇编文件，要求能被编译链接成可执行文件，并在安装有Linux操作系统的指定RISC-V硬件平台上加载并运行，以测试程序执行时间作为评价依据。生成的汇编程序应为64位，RISC-V体系结构。汇编代码要允许在较大地址空间内运行（即遵从GCC-mcmodel=mandany选项的约定)。

使用平台内置的GitLab（页面上方的导航栏）进行协同开发，并直接提交仓库地址（https地址）进行评测。

测评程序默认拉取master分支，如果需要选择其他分支进行测评，请在提交面板上输入仓库地址后，加空格输入分支名称，具体参照提交页面上的帮助。测评程序将拉取指定的分支进行测评。

测评程序会直接使用clang/clang++、javac或Cargo 对提交的项目代码进行编译，测评程序会自动扫描项目下的代码文件，测评程序支持的文件后缀包括：

C/C++语言实现，支持C++ 17标准：

l 头文件：h、hh、hpp、H、hxx

l 源文件：c、cpp、cp、cxx、cc、c++、C、CPP

请确保项目代码中只有一个main函数。



Java语言实现，支持openjdk24：

l 后缀：java

l 主类名：Compiler.java (主类不带包名)

测评程序会将编译好的class文件打包成jar格式并执行。若项目中包含res目录，则测评程序会将其中内容作为资源文件复制到jar包的根目录中。



Rust语言实现，支持rustc 1.85.0：

l 后缀：rs



【样例下载】

参赛队可访问2026年编译系统设计赛项目仓库下载公开用例



【输入形式】

学生将编译器统一命名为compiler。测试将通过如下命令将testcase.sy中的sysy2022语言的代码编译成testcase.s中的RISC-V汇编代码。（注意：测评时命令中的testcase.sy和testcase.s会被替换为相应文件的绝对路径，系统会保证编译器对这些文件有相应的读写权限,存放测试用例的目录是只读的）



l 功能测试：compiler -S -o testcase.s testcase.sy

l 性能测试：compiler -S -o testcase.s testcase.sy -O1



测试程序的具体输入输出方法请参考SysY运行时库。



【输出形式】

按如上要求将目标汇编代码生成结果输出至testcase.s文件中。



【评分要求】

评测过程分为功能测试和性能测试，按照功能得分、性能得分、性能测试总运行时间进行排名。具体评分要求请参考2026年全国大学生计算机系统能力大赛-编译系统实现赛技术方案



【测评结果说明】

测评结果按照CE、RE、TLE、WA、AC这一优先级进行显示。



总评：

CE：编译从gitlab上拉取的项目时发生了错误

RE：整个测评流程过程中发生了运行时错误。如拉取项目代码失败，某个用例发生了RE或TLE，未进行性能测试等

TLE：整个测评耗时超过了总限制时长，当前为30分

WA：存在测试用例WA

AC：测试通过



测试点：

CE：使用提交的编译器编译用例程序时发生了错误

RE：该测试点测评流程过程中发生了运行时错误。如测评程序找不到输出的.s文件、gcc汇编链接失败、测评程序没有找到计时函数的输出等

TLE：该测试点在生成汇编代码或执行汇编时超过了时间限制。

WA：最终在树莓派上的可执行文件的标准输出和期望输出 或 程序实际返回值和期望返回值不一样（注意：包括可执行程序执行失败）

AC：测试点测试通过，标准输出和期望输出一致、程序实际返回值和期望返回值一致



【优化违规行为包括但不限于】

针对特定用例名或函数名等的优化。

通过给定输出的方式蒙混过关。

尝试获得评测机、隐藏用例等信息。

其他试图获取不公平优势的行为。



【参考文件-附件】

SysY2022语言定义

The RISC-V Instruction Set Manual Volume I: Unprivileged ISA.

The RISC-V Instruction Set Manual Volume II: Privileged Architecture.

GCC RISC-V体系结构关于mcmodel选项的说明

参赛队需开发支持特定语言、面向ARM硬件平台或RISC-V硬件平台的综合性编译系统。



编译系统应基于C、C++、Java或Rust语言开发，并可在Ubuntu 24.04（64位）操作系统的x86 测评服务器完成编译。
ARM硬件平台：能够将符合自定义程序设计语言SysY2026的测评程序编译为 ARM 汇编语言程序（64位，ARMv8-A架构），并通过汇编链接后在基于赛灵思XCZU15EG（集成ARM Cortex-A53处理器）的Ubuntu 22.04操作系统上运行。
RISC-V硬件平台：能够将符合自定义程序设计语言SysY2026的测评程序编译为 64位RISC-V汇编语言程序。汇编代码需支持在较大地址空间内运行（即遵从GCC -mcmodel=medany选项约定）；并通过汇编链接后在64位 FPGA BOOM CPU软核上运行。

大赛指定的编译环境用于编译参赛队提交的编译系统源码，参数如下：
1. CPU：AMD64 架构
2. 内存：8GB
3. Docker 容器操作系统：Ubuntu 24.04
4. C/C++编译器：LLVM/Clang-18，编译选项：
C 语言：clang --std=c11 -O2 -lm
C++语言：clang++ --std=c++17 -O2 -lm
5. Java 平台与编译器：openjdk 24，主类名与编译选项：
主类名：Compiler.java （主类不带包名）
编译选项：javac -encoding utf-8
6. Rust 平台与编译器：rustc 1.85.0
编译选项：rustc -O2

大 赛 指 定 的 ARM 架 构 目 标 程 序 性 能 基 准 测 评 实 验 设 备 为
CG-FPGA15EG，主要参数如下：
1. CPU：赛灵思 XCZU15EG ARM Cortex-A53 MPCore ，4 核心（2 个隔离核用于测评），L1 数据缓存大小为 32 KB，4 路组相联，L1 指令缓存大小为 32 KB， 2 路组相联。 L2 缓存大小为 1MB， 由所有核心共享，16 路组相联。支持 ARMv8-A 64 位指令系统，支持 NEON 和单精度/双精度浮点计算。
2. 内存：4GB DDR4。
3. 操作系统：64 位，Ubuntu 22.04。
4. 汇编和链接器：gcc version 11.2.0（Ubuntu 11.20-19ubuntu1），编译与
链接命令：gcc -march=armv8-a（用于汇编和链接参赛队编译系统输出的汇编代码）。
5. 编译生成的参赛队编译器可执行文件统一命名为 compiler，参数如下：
功能测试：compiler testcase.sysy -S -o testcase.s
性能测试：compiler testcase.sysy -S -o testcase.s -O1
6. 为 提 升 性 能 测 评 的 计 时 精 度 ， 采 用 了 Linux CPU 隔 离 技 术
（isolcpus=2,3），目标程序运行在操作系统的核心 2 和核心 3 上。若参赛队编译器采用了多线程优化技术，请注意与此要求相匹配。

大 赛 指 定 的 RISC-V 架 构 目 标 程 序 性 能 基 准 测 评 实 验 设 备 为CG-FPGA15EG，主要参数如下：
1. CPU：BOOM v3（Berkeley Out-of-Order Machine Version 3）架构，RISC-V 64GC 指令集， 主频 50MHz， 乱序执行， 双发射， 超标量构架。
2. 内存：2GB。
3. 配备 FPU（浮点数运算单元，采用标准 IEEE754 格式）。初赛阶段，参赛队不得使用 SIMD 指令； 在决赛阶段， 参赛队可根据开放的 SIMD指令文档使用 SIMD 指令。
4. 参赛队的目标程序经过编译和链接后， 直接在 BOOM CPU 软核上运行，参赛队可通过 GDB+OpenOCD 对运行于软核上的目标程序进行调试。
5. 汇编和链接器：gcc version 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)，编译与链接命令：gcc -march=rv64gc（用于汇编和链接参赛队编译系统输出的汇编代码）。
6. 编译生成的参赛队编译器统一命名为 compiler，参数如下：
功能测试：compiler testcase.sysy -S -o testcase.s
性能测评：compiler testcase.sysy -S -o testcase.s -O1
