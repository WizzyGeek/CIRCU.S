from __future__ import annotations

from enum import Enum, IntEnum, StrEnum
import typing as t

class Operator(StrEnum):
    AND  = "&"
    OR   = "|"
    XOR  = "^"
    ASSIGN = "="
    # XNOR = "~^" # Later
    # NAND = "~&"
    # NOR  = "~|"

op_precedence = {
    "&": 50,
    "|": 50,
    "^": 50,
    "=": 10
}

class UnraryOperator(StrEnum):
    NOT = "!"


AND_LUT = [
    [0, 0, 0],
    [0, 1, 2],
    [0, 2, 0]
]

OR_LUT = [
    [0, 1, 2],
    [1, 1, 1],
    [2, 1, 2]
]

NOT_LUT = [
    1, 0, 2
]

XOR_LUT = [
    [0, 1, 2],
    [1, 0, 2],
    [2, 2, 2]
]

class Operand:
    __slots__ = ("index")
    index: int

    def __init__(self, index: int):
        self.index = index

class BitOperand:
    __slots__ = ("index", "bidx")
    index: int
    bidx: int

    def __init__(self, index: int, bidx: int):
        self.index = index
        self.bidx  = bidx

class Expression:
    le: Expression | Operand | BitOperand | UnraryOp
    re: Expression | Operand | BitOperand | UnraryOp
    op: Operator

    __slots__ = ("le", "re", "op")

class UnraryOp:
    operand: Expression | Operand | BitOperand | UnraryOp
    op: Operator

    __slots__ = ("operand", "op")


class TokenType(Enum):
    NEWLINE = 0
    LPARA   = 1
    RPARA   = 2
    AND     = 3
    OR      = 4
    XOR     = 5
    NOT     = 6
    IDENT   = 7
    OUTP    = 8
    PERS    = 9
    INP     = 10
    VEC     = 11
    LRECT   = 12
    RRECT   = 13
    NUM     = 14
    ASSIGN  = 15
    EOF     = 16
    LET     = 17

class Token:
    type: TokenType
    value: str

    __slots__ = ("type", "value")

    def __init__(self, type: TokenType, value: str) -> None:
        self.type = type
        self.value = value

single_char_lut = {
    "(": TokenType.LPARA,
    "[": TokenType.LRECT,
    "]": TokenType.RRECT,
    ")": TokenType.RPARA,
    "\n": TokenType.NEWLINE,
    ";": TokenType.NEWLINE,
    "&": TokenType.AND,
    "^": TokenType.XOR,
    "|": TokenType.OR,
    "!": TokenType.NOT,
    "=": TokenType.ASSIGN,
}

kw_lut = {
    "persistent": TokenType.PERS,
    "vector": TokenType.VEC,
    "output": TokenType.OUTP,
    "input": TokenType.INP,
    "let":  TokenType.LET
}

class Tokeniser:
    source: str
    curr: int
    slen: int
    open: bool

    def __init__(self, source: str) -> None:
        self.source = source
        self.slen = len(source)
        self.curr = 0
        self.open = self.curr < self.slen
        self._yield_next = None

    def next_token(self) -> Token:
        if self._yield_next is not None:
            try: return self._yield_next
            finally: self._yield_next = None

        s = self.source
        sl = self.slen
        c = self.curr
        try:
            while c < sl and (s[c].isspace() and s[c] != "\n"):
                c += 1

            if c == sl:
                return Token(TokenType.EOF, "")

            buf = s[c]
            if buf in single_char_lut:
                return Token(single_char_lut[buf], buf)

            c += 1
            while c < sl and not (s[c].isspace() or s[c] == "[" or s[c] == "]"):
                buf += s[c]
                c += 1
            c -= 1

            if buf in kw_lut:
                return Token(kw_lut[buf], buf)

            if buf.isdecimal():
                return Token(TokenType.NUM, buf)

            if buf.isidentifier():
                return Token(TokenType.IDENT, buf)

            raise ValueError(f"Invalid word at index {c}: {buf!r}")
        finally:
            self.curr = c + 1
            self.open = c < sl

    def undo(self, tok: Token) -> None:
        self._yield_next = tok

class VariableType(IntEnum):
    INTER = 0
    OUTP  = 1
    INP   = 2
    PERS  = 3

# s -> v
# v? index
class Variable:
    vector: int
    index: int
    type: VariableType

    def __init__(self, vector: int, index: int, type: VariableType)-> None:
        self.vector = vector
        self.index = index
        self.type = type

    def make_operand(self):
        return Operand(self.index)

    def make_bit_operand(self, bidx: int):
        if bidx >= self.vector:
            raise ValueError(f"Out of bounds index {bidx} for vector at index {self.index}")
        return BitOperand(self.index, bidx)

# class Circuit:

variable_type_lut = {
    TokenType.OUTP: VariableType.OUTP,
    TokenType.INP: VariableType.INP,
    TokenType.PERS: VariableType.PERS,
    TokenType.LET: VariableType.INTER
}

class Parser:
    variables: dict[str, Variable]
    body: list[Expression]

    def __init__(self) -> None:
        self.variables = {}
        self.vlist = []
        self.body = []
        self._curr_idx = 0

    # shunt yard algorithm
    def parse(self, tok: Tokeniser) -> None:
        comb = []
        aux = []
        while tok.open:
            t = tok.next_token()
            if t.type in variable_type_lut:
                tok.undo(t)
                self.parse_decl(tok)
                continue

            if t.type == TokenType.IDENT:
                comb.append(self.variables[t.value].make_operand())
            ...

    def parse_decl(self, tok: Tokeniser) -> None:
        t = tok.next_token()
        mod = t.type
        if mod not in variable_type_lut:
            raise ValueError(f"Expected declrator token, found: {t.type}({t.value!r})")

        t = tok.next_token()
        vector: int = 0
        if t.type == TokenType.VEC:
            vector = 1

        if vector != 1 and t.type != TokenType.IDENT:
            raise ValueError(f"Expected modifier or identifier after declrator, found: {t.type}({t.value!r})")

        iden = ""
        if vector == 0:
            iden = t.value
        if vector == 1:
            t = tok.next_token()
            if t.type != TokenType.IDENT:
                raise ValueError(f"Expected identifier after modifier, found:  {t.type}({t.value!r})")
            iden = t.value

            t = tok.next_token()
            if t.type != TokenType.LRECT:
                raise ValueError(f"Expected vector size to be specifed in declration for identifier: {iden}")

            t = tok.next_token()
            if t.type != TokenType.NUM:
                raise ValueError(f"Vector size not numeric in declration for identifier: {iden}")
            else:
                vector = int(t.value)

            t = tok.next_token()
            if t.type != TokenType.RRECT:
                raise ValueError(f"Unclosed parantheses '[' in declration for identifier: {iden}")

        if iden in self.variables:
            raise ValueError(f"Cannot redeclare identifier {iden}")

        self.variables[iden] = k = Variable(vector, self._curr_idx, variable_type_lut[mod])
        self.vlist.append(k)
        self._curr_idx += 1

# t = Tokeniser("""output vector a[3];""")
# p = Parser()

# p.parse_decl(t)
# print(p.variables)
# while t.open:
#     tok = t.next_token()
#     print(tok.type, repr(tok.value))