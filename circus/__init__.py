from __future__ import annotations

from enum import Enum, IntEnum, StrEnum
import typing as t

from textwrap import indent


class Operator(StrEnum):
    AND  = "&"
    OR   = "|"
    XOR  = "^"
    ASSIGN = "="
    # XNOR = "~^" # Later
    # NAND = "~&"
    # NOR  = "~|"

    @property
    def precedence(self):
        return op_precedence[self]

    def can_stack(self, stack: list[WorkingOp]):
        return (not stack) or (stack[-1] == "(") or (stack[-1].precedence < self.precedence)


class UnraryOperator(StrEnum):
    NOT = "!"

    @property
    def precedence(self):
        return 100

    def can_stack(self, stack: list[WorkingOp]):
        return True

WorkingOp = Operator | UnraryOperator | t.Literal["("]

op_precedence = {
    Operator.AND: 50,
    Operator.OR: 50,
    Operator.XOR: 50,
    Operator.ASSIGN: 10
}

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

class Lvalue:
    pass

class Rvalue:
    pass

class Operand(Lvalue, Rvalue):
    __slots__ = ("index")
    index: int

    def __init__(self, index: int):
        self.index = index

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(index={self.index})"

class BitOperand(Lvalue, Rvalue):
    __slots__ = ("index", "bidx")
    index: int
    bidx: int

    def __init__(self, index: int, bidx: int):
        self.index = index
        self.bidx  = bidx

    def __repr__(self) -> str:
        i = indent(f'\nindex={self.index},\nbidx={self.bidx}', "    ")
        return f"{self.__class__.__name__}({i}\n)"

class Expression(Rvalue):
    le: Node
    re: Node
    op: Operator

    __slots__ = ("le", "re", "op")

    def __init__(self, re: Node, le: Node, op: Operator) -> None:
        if op == Operator.ASSIGN and not isinstance(le, Lvalue):
            raise ValueError("Cannot assign to a non lvalue")
        self.le = le
        self.re = re
        self.op = op

    def __repr__(self) -> str:
        i = indent(f'\nop={self.op!r},\nle={self.le},\nre={self.re}', "    ")
        return f"{self.__class__.__name__}({i}\n)"

class UnraryOp(Rvalue, Lvalue):
    operand: Node
    op: UnraryOperator

    __slots__ = ("operand", "op")

    def __init__(self, operand: Node, op: UnraryOperator) -> None:
        self.operand = operand
        self.op = op

    def __repr__(self) -> str:
        i = indent(f'\nop={self.op},\noperand={self.operand}', "    ")
        return f"{self.__class__.__name__}({i}\n)"


class Constant(Rvalue):
    value: int

    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.value})"


Node = Expression | Operand | BitOperand | UnraryOp | Constant

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

op_token_precedence = {
    single_char_lut[c]: prec for c, prec in op_precedence.items()
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
            while c < sl and not (s[c].isspace() or s[c] in single_char_lut):
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

    def __repr__(self) -> str:
        i = indent(f'\nvector={self.vector},\nindex={self.index},\ntype={self.type!r}', "    ")
        return f"{self.__class__.__name__}({i}\n)"

# class Circuit:

variable_type_lut = {
    TokenType.OUTP: VariableType.OUTP,
    TokenType.INP: VariableType.INP,
    TokenType.PERS: VariableType.PERS,
    TokenType.LET: VariableType.INTER
}

class Parser:
    variables: dict[str, Variable]
    vlist: list[Variable]
    body: list[Node]
    _curr_idx: int

    def __init__(self) -> None:
        self.variables: dict[str, Variable] = {}
        self.vlist: list[Variable] = []
        self.body = []
        self._curr_idx = 0

    def __repr__(self) -> str:
        i = indent(f'\nvariables={self.variables},\nbody={self.body}', "    ")
        return f"{self.__class__.__name__}({i}\n)"

    # shunt yard algorithm
    def parse(self, tok: Tokeniser) -> None:
        comb: list[Node] = []
        aux: list[WorkingOp] = []
        while tok.open:
            t = tok.next_token()
            if t.type in variable_type_lut:
                tok.undo(t)
                self.parse_decl(tok)
                comb.append(self.vlist[-1].make_operand())
            elif t.type == TokenType.IDENT:
                t2 = tok.next_token()
                if t2.type == TokenType.LRECT:
                    try:
                        t2 = tok.next_token()
                        assert t2.type == TokenType.NUM
                        bidx = int(t2.value)
                        assert tok.next_token().type == TokenType.RRECT
                    except AssertionError as err:
                        raise ValueError("Invalid indexing syntax") from err
                    if not self.variables[t.value].vector:
                        raise ValueError("Cannot index non vector operand")
                    comb.append(self.variables[t.value].make_bit_operand(bidx))
                else: tok.undo(t2)

                try:
                    comb.append(self.variables[t.value].make_operand())
                except KeyError as err:
                    raise ValueError(f"Identifier {t.value!r} not declared before use") from err
            elif t.type in op_token_precedence:
                op = Operator(t.value)
                while not op.can_stack(aux):
                    w = aux.pop()
                    if isinstance(w, Operator):
                        comb.append(Expression(comb.pop(), comb.pop(), w))
                    elif isinstance(w, UnraryOperator):
                        comb.append(UnraryOp(comb.pop(), w))
                    else:
                        raise RuntimeError # Unreachable
                aux.append(op)
            elif t.type == TokenType.LPARA:
                aux.append("(")
            elif t.type == TokenType.RPARA:
                while aux and ((wop := aux.pop()) != "("):
                    try:
                        if isinstance(wop, Operator):
                            comb.append(Expression(comb.pop(), comb.pop(), wop))
                        elif isinstance(wop, UnraryOperator):
                            comb.append(UnraryOp(comb.pop(), wop))
                    except IndexError:
                        raise ValueError("Invalid expression")
            elif t.type == TokenType.NUM:
                comb.append(Constant(int(t.value)))
            elif t.type == TokenType.NEWLINE or t.type == TokenType.EOF:
                while aux:
                    # print(aux, comb)
                    wop = aux.pop()
                    try:
                        if isinstance(wop, Operator):
                            comb.append(Expression(comb.pop(), comb.pop(), wop))
                        elif isinstance(wop, UnraryOperator):
                            comb.append(UnraryOp(comb.pop(), wop))
                        else: raise ValueError("Expression terminated with unbalanced parantheses")
                    except IndexError:
                        raise ValueError("Invalid Expression")
                if len(comb) > 1:
                    print(comb)
                    raise ValueError("Invalid Expression, irreducible.")
                # Could have assign inside! this should not be optimised here
                # if isinstance(comb[0], Expression) and comb[0].op == Operator.ASSIGN:
                if comb:
                    self.body.append(comb[0])
                    comb.clear()
            else:
                print("Unexpected token", t, t.type, t.value)


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


if __name__ == "__main__":
    t = Tokeniser(
        """
        persistent k;
        output q;
        input t;
        input p;
        k = p | (t ^ k);
        q = k;
        """
    )
    p = Parser()

    # p.parse_decl(t)
    # print(p.variables)
    # while t.open:
    #     tok = t.next_token()
    #     print(tok.type, repr(tok.value))
    p.parse(t)
    from pprint import pprint
    pprint(p)
