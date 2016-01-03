# Cuda Samples

#### [01_HelloWorld](https://github.com/ufukomer/cuda-samples/tree/master/01_HelloWorld)
```
H
E
L
L
O

W
O
R
L
D
!
Press any key to continue . . .
```

#### [02_VectorAdd](https://github.com/ufukomer/cuda-samples/tree/master/02_VectorAdd)
```
c[0] = 2
c[1] = 4
c[2] = 6
c[3] = 8
c[4] = 10
c[5] = 12
c[6] = 14
c[7] = 16
c[8] = 18
c[9] = 20
c[10] = 22
c[11] = 24
c[12] = 26
c[13] = 28
c[14] = 30
c[15] = 32
c[16] = 34
c[17] = 36
c[18] = 38
c[19] = 40
c[20] = 42
c[21] = 44
c[22] = 46
c[23] = 48
c[24] = 50
c[25] = 52
c[26] = 54
c[27] = 56
c[28] = 58
c[29] = 60
c[30] = 62
c[31] = 64
c[32] = 66
c[33] = 68
c[34] = 70
c[35] = 72
c[36] = 74
c[37] = 76
c[38] = 78
c[39] = 80
c[40] = 82
c[41] = 84
c[42] = 86
c[43] = 88
c[44] = 90
c[45] = 92
c[46] = 94
c[47] = 96
c[48] = 98
c[49] = 100
c[50] = 102
c[51] = 104
c[52] = 106
c[53] = 108
c[54] = 110
c[55] = 112
c[56] = 114
c[57] = 116
c[58] = 118
c[59] = 120
c[60] = 122
c[61] = 124
c[62] = 126
c[63] = 128
c[64] = 0
Press any key to continue . . .
```

#### [03_MatrixAdd](https://github.com/ufukomer/cuda-samples/tree/master/03_MatrixAdd)
```
c[0] =  13
c[1] =  15
c[2] =  5
c[3] =  9
c[4] =  8
c[5] =  8
c[6] =  5
c[7] =  18
c[8] =  5
c[9] =  8
c[10] =  3
c[11] =  5
Press any key to continue . . .
```

#### [04_MatrixMult](https://github.com/ufukomer/cuda-samples/tree/master/04_MatrixMult)
```
c[0] =  5
c[1] =  4
c[2] =  4
c[3] =  5
Press any key to continue . . .
```

#### [05_BlockId](https://github.com/ufukomer/cuda-samples/tree/master/05_BlockId)
```
Hello world! I'm a thread in block 11
Hello world! I'm a thread in block 13
Hello world! I'm a thread in block 2
Hello world! I'm a thread in block 4
Hello world! I'm a thread in block 0
Hello world! I'm a thread in block 1
Hello world! I'm a thread in block 3
Hello world! I'm a thread in block 7
Hello world! I'm a thread in block 9
Hello world! I'm a thread in block 10
Hello world! I'm a thread in block 6
Hello world! I'm a thread in block 8
Hello world! I'm a thread in block 12
Hello world! I'm a thread in block 14
Hello world! I'm a thread in block 15
Hello world! I'm a thread in block 5
That's all!
Press any key to continue . . .
```

#### [06_RaceCondition](https://github.com/ufukomer/cuda-samples/tree/master/06_RaceCondition)
```
1000000 total threads in 1000 blocks writing into 10 array elements
{ 500 500 506 506 506 506 506 506 506 506 }
Time elapsed = 1.11213 ms
Press any key to continue . . .
```

#### [07_AtomicAdd](https://github.com/ufukomer/cuda-samples/tree/master/07_AtomicAdd)
```
1000000 total threads in 1000 blocks writing into 100 array elements
{ 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10
000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 1000
0 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10
000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 1000
0 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 }
Time elapsed = 1.1801 ms
Press any key to continue . . .
```

#### [08_Reduction](https://github.com/ufukomer/cuda-samples/tree/master/08_Reduction)
```
d[0]: 80
d[1]: 21
d[2]: 98
Press any key to continue . . .
```

#### [09_HistogramCalculate](https://github.com/ufukomer/cuda-samples/tree/master/09_HistogramCalculate)
```
Out[0]: 12
Out[1]: 11
Out[2]: 7
Out[3]: 7
Out[4]: 4
Out[5]: 12
Out[6]: 13
Out[7]: 15
Out[8]: 20
Out[9]: 15
Out[10]: 9
Out[11]: 8
Out[12]: 14
Out[13]: 13
Out[14]: 13
Out[15]: 7
Out[16]: 9
Out[17]: 15
Out[18]: 10
Out[19]: 14
Out[20]: 12
Out[21]: 11
Out[22]: 6
Out[23]: 10
Out[24]: 11
Out[25]: 7
Out[26]: 11
Out[27]: 11
Out[28]: 13
Out[29]: 10
Out[30]: 10
Out[31]: 11
Out[32]: 7
Out[33]: 14
Out[34]: 18
Out[35]: 7
Out[36]: 14
Out[37]: 13
Out[38]: 8
Out[39]: 8
Out[40]: 11
Out[41]: 6
Out[42]: 7
Out[43]: 14
Out[44]: 6
Out[45]: 13
Out[46]: 7
Out[47]: 19
Out[48]: 12
Out[49]: 10
Out[50]: 7
Out[51]: 7
Out[52]: 8
Out[53]: 10
Out[54]: 4
Out[55]: 10
Out[56]: 10
Out[57]: 10
Out[58]: 10
Out[59]: 10
Out[60]: 8
Out[61]: 12
Out[62]: 7
Out[63]: 12
Out[64]: 11
Out[65]: 12
Out[66]: 14
Out[67]: 7
Out[68]: 7
Out[69]: 3
Out[70]: 7
Out[71]: 11
Out[72]: 7
Out[73]: 13
Out[74]: 9
Out[75]: 8
Out[76]: 19
Out[77]: 5
Out[78]: 8
Out[79]: 7
Out[80]: 8
Out[81]: 8
Out[82]: 5
Out[83]: 7
Out[84]: 17
Out[85]: 15
Out[86]: 14
Out[87]: 11
Out[88]: 9
Out[89]: 13
Out[90]: 8
Out[91]: 12
Out[92]: 13
Out[93]: 10
Out[94]: 13
Out[95]: 7
Out[96]: 8
Out[97]: 13
Out[98]: 5
Out[99]: 10
Press any key to continue . . .
```

#### [10_MaxValue](https://github.com/ufukomer/cuda-samples/tree/master/10_MaxValue)
```
Max: 55
Max: 8
Max: 87
Press any key to continue . . .
```

#### Technical Specifications

![tech-specifications](http://blog.cuvilib.com/wp-content/uploads/2010/06/fermi_vs_old.png)
