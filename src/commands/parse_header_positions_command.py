from bs4 import BeautifulSoup

from src.commands.command import Command


class ParseHeaderPositionsCommand(Command):
    def handle(self, context: dict) -> bool:
        with open('../resources/headers.html', 'r') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        rows = soup.find_all('tr')

        data_dict = {}

        for row in rows:
            cols = row.find_all('td')

            if len(cols) <= 1:
                continue

            from_column = int(cols[0].text.strip())
            column_name = cols[1].text.strip()
            column_size = cols[2].text.strip()
            to_column = (int(from_column) + int(column_size))

            data_dict[column_name] = {'from': from_column, 'to': to_column}

        context['header_positions'] = data_dict
        return True
