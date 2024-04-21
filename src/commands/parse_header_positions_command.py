from bs4 import BeautifulSoup

from src.commands.command import Command


class ParseHeaderPositionsCommand(Command):
    def handle(self, context: dict) -> bool:
        with open('../resources/headers.html', 'r') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        rows = soup.find_all('tr')

        data_dict = {}

        last_name = None

        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1:
                col_index = int(cols[0].text.strip())
                col_name = cols[1].text.strip()

                if col_name not in data_dict:
                    data_dict[col_name] = {'from': col_index}

                if last_name and last_name != col_name:
                    data_dict[last_name]['to'] = col_index - 1

                last_name = col_name

        with open('../resources/generated/dataset/LLCP2022.ASC ', 'r') as file:
            content = file.read()
            data_dict[last_name]['to'] = len(content.splitlines().pop())

        context['header_positions'] = data_dict
        return True
