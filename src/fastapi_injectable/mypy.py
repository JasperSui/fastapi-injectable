"""Mypy plugin for fastapi-injectable.

This plugin provides type checking support for the @injectable decorator,
making Annotated[Type, Depends(...)] parameters optional from the caller's perspective.

Based on patterns from:
- https://github.com/python/mypy/tree/master/mypy/plugins
- https://github.com/pydantic/pydantic/blob/main/pydantic/mypy.py
- https://github.com/dropbox/sqlalchemy-stubs/blob/master/sqlmypy.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_OPT,
    ARG_POS,
    ArgKind,
    Argument,
    Decorator,
    Expression,
    FuncDef,
    NameExpr,
)
from mypy.plugin import FunctionContext, FunctionSigContext, Plugin
from mypy.types import CallableType, RawExpressionType, Type, UnboundType

if TYPE_CHECKING:
    from collections.abc import Callable


INJECTABLE_DECORATOR_FULLNAMES = {
    "fastapi_injectable.decorator.injectable",
    "src.fastapi_injectable.decorator.injectable",
}


class FastApiInjectablePlugin(Plugin):
    """Mypy plugin for fastapi-injectable.

    This plugin modifies function signatures to make dependency-injected parameters
    optional from the caller's perspective while maintaining type safety.
    """

    def __init__(self, options: Any) -> None:  # noqa: ANN401
        """Initialize the plugin."""
        super().__init__(options)

    def get_function_signature_hook(self, fullname: str) -> Callable[[FunctionSigContext], CallableType] | None:
        """Hook for modifying function signatures.

        This is called for every function that mypy analyzes.
        We only process functions decorated with @injectable.
        """
        sym_table_node = self.lookup_fully_qualified(fullname)
        if not sym_table_node or not isinstance(sym_table_node.node, Decorator):
            return None

        if any(
            name_expr.fullname in INJECTABLE_DECORATOR_FULLNAMES
            for name_expr in sym_table_node.node.decorators
            if isinstance(name_expr, NameExpr)
        ):
            return self._process_injectable_function_signature

        return None

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        """Hook for modifying method/function calls.

        This handles injectable(func) direct calls.
        """
        if fullname in INJECTABLE_DECORATOR_FULLNAMES:
            return self._process_injectable_call
        return None

    def _process_injectable_call(self, ctx: FunctionContext) -> Type:
        """Process injectable(func) direct calls.

        This modifies the return type to make dependency parameters optional.
        """
        if not ctx.arg_types or not ctx.arg_types[0]:
            return ctx.default_return_type

        # Get the function being passed to injectable()
        original_callable_type = ctx.arg_types[0][0]
        if not isinstance(original_callable_type, CallableType):
            return ctx.default_return_type

        # Check if we have access to the function definition
        if not ctx.args or not ctx.args[0] or not isinstance(ctx.args[0][0], NameExpr):
            return ctx.default_return_type

        func_node = ctx.args[0][0].node
        if not isinstance(func_node, FuncDef):
            return ctx.default_return_type

        # Since we can't get the Annotated[Type, Depends(...)] from the FuncDef node anymore (it's from deserialization)
        # We can only treat all arguments as optional for now,
        # But the best option is to get the Annotated[Type, Depends(...)] from the FuncDef node,
        # Then check if the argument is Annotated[Type, Depends(...)], if so, make it optional, otherwise, keep it as is.  # noqa: E501
        modified_arg_kinds = []
        for arg_kind in original_callable_type.arg_kinds:
            if arg_kind == ARG_POS:
                modified_arg_kinds.append(ARG_OPT)
            elif arg_kind == ARG_NAMED:
                modified_arg_kinds.append(ARG_NAMED_OPT)
            else:
                modified_arg_kinds.append(arg_kind)

        return original_callable_type.copy_modified(arg_kinds=modified_arg_kinds)

    def _process_injectable_function_signature(self, ctx: FunctionSigContext) -> CallableType:
        """Process the signature of an @injectable decorated function.

        This makes Annotated[Type, Depends(...)] parameters optional.
        """
        original_signature = ctx.default_signature
        decorated_func_def_node: FuncDef = ctx.context.callee.node.func  # type: ignore[attr-defined]

        # Process the function signature using shared logic
        modified_arg_kinds = self._modify_arg_kinds_for_dependencies(
            decorated_func_def_node.arguments, original_signature.arg_kinds
        )

        # Return modified signature if we made changes
        if modified_arg_kinds != original_signature.arg_kinds:
            return original_signature.copy_modified(arg_kinds=modified_arg_kinds)

        return original_signature

    def _modify_arg_kinds_for_dependencies(
        self, arguments: list[Argument], original_arg_kinds: list[ArgKind]
    ) -> list[ArgKind]:
        """Modify argument kinds to make dependency-injected parameters optional.

        Args:
            arguments: List of function argument nodes
            original_arg_kinds: Original argument kinds from the callable type

        Returns:
            Modified list of argument kinds
        """
        modified_arg_kinds = list(original_arg_kinds)

        # Process each argument
        for i, arg_node in enumerate(arguments):
            # Safety check for bounds
            if i >= len(original_arg_kinds):
                break

            # Check if this argument has a dependency annotation
            if arg_node.type_annotation and self._is_fastapi_depends_annotation(arg_node.type_annotation):  # type: ignore[arg-type]
                current_kind = original_arg_kinds[i]

                # Make required arguments optional
                if current_kind == ARG_POS:  # Required positional
                    modified_arg_kinds[i] = ARG_OPT  # Make optional positional
                elif current_kind == ARG_NAMED:  # Required keyword-only
                    modified_arg_kinds[i] = ARG_NAMED_OPT  # Make optional keyword-only

        return modified_arg_kinds

    def _is_fastapi_depends_annotation(self, type_expr: Expression) -> bool:
        """Check if a type annotation is Annotated[Type, Depends(...)].

        This method handles the complex AST traversal to identify dependency annotations.
        """
        if not isinstance(type_expr, UnboundType):  # type: ignore[unreachable]
            return False

        # type_expr -> Annotated[Type, Depends(...)], type_expr.args -> (Type, None)
        if type_expr.name != "Annotated" or len(type_expr.args) < 2:  # type: ignore[unreachable]  # noqa: PLR2004
            return False

        # Suggestion: use Depends[...] instead of Depends(...) is where the magic happens
        # https://github.com/python/mypy/blob/a8ec8939ce5a8ba332ec428bec8c4b7ef8c42344/mypy/fastparse.py#L1958-L1969
        # Since there is no way to get the annotation metadata from mypy as of now, ref: https://github.com/python/mypy/pull/9625
        # So the metadata will be a None in type_expr.args, hopefully, the note will indicate that the annotation is a Depends annotation  # noqa: E501
        # so we can rely on the note to determine if the annotation is a Depends annotation for now.
        return any(
            arg.note == "Suggestion: use Depends[...] instead of Depends(...)"
            for arg in type_expr.args
            if isinstance(arg, RawExpressionType)
        )


def plugin(version: str) -> type[Plugin]:  # noqa: ARG001
    """Mypy plugin entry point."""
    return FastApiInjectablePlugin
