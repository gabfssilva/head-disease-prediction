from commands.command import Command


def remove_nan_on_target(context: dict) -> bool:
    context['raw_dataset'] = context['raw_dataset'].dropna(subset=['CVDINFR4'])
    return True
