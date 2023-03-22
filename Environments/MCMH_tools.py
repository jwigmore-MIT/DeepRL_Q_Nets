from functools import singledispatch

@singledispatch
def keys_to_strings(ob):
    return ob

@keys_to_strings.register
def _handle_dict(ob: dict):
    return {str(k): keys_to_strings(v) for k, v in ob.items()}

@keys_to_strings.register
def _handle_list(ob: list):
    return [keys_to_strings(v) for v in ob]

@singledispatch
def keys_to_ints(ob):
    return ob

@singledispatch
def keys_to_tup(ob):
    return ob

@keys_to_tup.register
def _handle_dict(ob: dict):
    return {tuple(k): v for k, v in ob.items}

@keys_to_ints.register
def _handle_dict(ob: dict):
    return {int(k): keys_to_ints(v) for k,v in ob.items()}

@keys_to_ints.register
def _handle_list(ob: list):
    return [keys_to_ints(v) for v in ob]


