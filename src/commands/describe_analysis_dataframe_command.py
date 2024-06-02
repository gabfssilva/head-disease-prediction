from src.commands.command import Command

def describe(context: dict) -> bool:
    print("Statistical Summary:")
    print(context['analysis_dataframe'].describe())
    return True
