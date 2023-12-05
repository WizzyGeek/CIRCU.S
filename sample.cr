persistent x
input d
input wen
x = (x & ~wen) | (d & wen)
output o = x
