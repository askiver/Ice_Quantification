turbine_dict = {}

for idx, value in enumerate(["07", "21", "41"]):
    turbine_dict[chr(65 + idx)] = value
    turbine_dict[value] = chr(65 + idx)

def replace_turbine_number(value):
    # If value is a single character or single number, return single term back
    if len(value) < 3:
        return turbine_dict.get(value)
    # If not, change the path to a correct term
    return value.replace("07", "A").replace("21", "B").replace("41", "C")
