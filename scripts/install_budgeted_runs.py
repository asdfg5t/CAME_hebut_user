"""Add independent 12h/24h configurations without changing any existing full-run entry."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / '.run'
    directory.mkdir(exist_ok=True)
    for hours in (24, 12):
        path = directory / ('CAME_Budgeted_' + str(hours) + 'h.run.xml')
        if path.exists():
            print('Preserved: ' + str(path))
            continue
        component = ET.Element('component', {'name': 'ProjectRunConfigurationManager'})
        config = ET.SubElement(component, 'configuration', {'default': 'false', 'name': 'CAME_Budgeted_' + str(hours) + 'h',
                               'type': 'PythonConfigurationType', 'factoryName': 'Python', 'singleton': 'true'})
        ET.SubElement(config, 'module', {'name': 'came-repro'})
        options = {'INTERPRETER_OPTIONS': '-u', 'PARENT_ENVS': 'true',
                   'SDK_HOME': str(ROOT / '.venv-came39/Scripts/python.exe').replace('\\', '/'),
                   'WORKING_DIRECTORY': '$PROJECT_DIR$', 'IS_MODULE_SDK': 'false',
                   'ADD_CONTENT_ROOTS': 'true', 'ADD_SOURCE_ROOTS': 'true',
                   'SCRIPT_NAME': '$PROJECT_DIR$/scripts/run_budgeted_pipeline.py',
                   'PARAMETERS': '--profile "$PROJECT_DIR$/configs/budgeted_' + str(hours) + 'h.json"',
                   'SHOW_COMMAND_LINE': 'true', 'EMULATE_TERMINAL': 'false', 'MODULE_MODE': 'false'}
        for name, value in options.items():
            ET.SubElement(config, 'option', {'name': name, 'value': value})
        envs = ET.SubElement(config, 'envs')
        for name, value in {'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}.items():
            ET.SubElement(envs, 'env', {'name': name, 'value': value})
        ET.SubElement(config, 'method', {'v': '2'})
        ET.indent(component)
        ET.ElementTree(component).write(path, encoding='utf-8', xml_declaration=True)
        print(path)


if __name__ == '__main__':
    main()
