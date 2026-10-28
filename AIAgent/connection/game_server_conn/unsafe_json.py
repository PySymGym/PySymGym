BareObject = type("BareObject", (), {})


def obj_from_dict(data: dict | list | str | float) -> BareObject:
    if isinstance(data, list):
        return [obj_from_dict(item) for item in data]
    if isinstance(data, dict):
        inner_dict = {
            field_name: obj_from_dict(field_raw)
            for field_name, field_raw in data.items()
        }
        bare_obj = BareObject()
        bare_obj.__dict__ = inner_dict
        return bare_obj
    return data
