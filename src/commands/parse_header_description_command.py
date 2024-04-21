import json

from bs4 import BeautifulSoup

from src.commands.command import Command


class ParseHeaderDescriptionsCommand(Command):
    def handle(self, context: dict) -> bool:
        with open('../resources/dataset header description.html', 'r', encoding='windows-1252') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        tables = soup.find_all('table', {'class': 'table'})

        variables_data_correct = {}

        for table in tables[1:]:
            rows = table.find_all('tr')
            if not rows:
                continue

            header_info = rows[0].get_text(separator="|", strip=True)
            details = header_info.split('|')
            if len(details) < 6:
                continue

            label = details[0].split(':')[1].strip() if 'Label:' in details[0] else None
            question = details[-1].split(':')[1].strip() if 'Question:' in details[-1] else None
            variable_name = details[6].split(':')[1].strip() if 'SAS\xa0Variable\xa0Name:' in details[6] else None

            possible_answers = []
            for data_row in rows[2:]:  # Skipping header rows
                cells = data_row.find_all('td')
                if len(cells) < 2:
                    continue
                value = cells[0].text.strip()
                description = cells[1].text.strip()
                possible_answers.append({
                    "description": description,
                    "value": value
                })

            if variable_name:
                variables_data_correct[variable_name] = {
                    "label": label,
                    "question": question,
                    "possible_answers": possible_answers
                }

        context['header_descriptions'] = json.loads(
            json
            .dumps(
                obj=variables_data_correct,
                indent=0,
                ensure_ascii=False
            )
            .replace(' ', ' ')
        )

        return True
