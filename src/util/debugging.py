from inspect import signature
from typing import Any, Callable, Union

from functools import wraps, partial
from jax.interpreters.partial_eval import JaxprTracer, Trace
from jax import jit, grad, vjp
from jax.tree_util import tree_map

__all__ = ["set_debug_reprs", "djit"]


def _id_str(obj: Any) -> str:
    id_ = hex(id(obj) & 0xFFFFF)[-5:]
    if hasattr(obj, "_name"):
        return f"{obj._name}:{id_}"
    return id_


def set_debug_reprs():
    def trace_repr(self: Trace):
        level = f"{self.level}/{self.sublevel}"
        return f"{self.__class__.__name__}({_id_str(self)}:{level})"

    def jaxpr_tracer_repr(self: JaxprTracer):
        trace = self._trace
        trace_id = _id_str(trace)
        return f"Tracer<{trace_id}::{_id_str(self)}>"

    Trace.__repr__ = trace_repr
    JaxprTracer.__repr__ = jaxpr_tracer_repr


def name_tracer(function_name, argument_name, tracer):
    if not isinstance(tracer, JaxprTracer):
        return tracer
    tracer._name = argument_name
    tracer._trace._name = function_name
    print(tracer)
    return tracer


def function_name(fun: Callable) -> str:
    if hasattr(fun, "__name__"):
        return fun.__name__
    if hasattr(fun, "func"):
        return function_name(fun.func)
    return "NAMELESS"


def name_all_tracers(fun):
    s = signature(fun)

    @wraps(fun)
    def new_fun(*args, **kwargs):
        bound_arguments = s.bind(*args, **kwargs)
        for name, value in bound_arguments.arguments.items():
            tree_map(partial(name_tracer, function_name(fun), name), value)
        return fun(*args, **kwargs)

    return new_fun


def djit(fun: Callable, *args, **kwargs):
    new_fun = name_all_tracers(fun)
    return jit(new_fun, *args, **kwargs)


def dgrad(fun, *args, **kwargs):
    new_fun = name_all_tracers(fun)
    return grad(new_fun, *args, **kwargs)


def dvjp(fun, *args, **kwargs):
    new_fun = name_all_tracers(fun)
    return vjp(new_fun, *args, **kwargs)
