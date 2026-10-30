"""Write shared per-run SDK overrides; preserve the user's existing .idea settings."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = (
    ('00_Run_Main_Pipeline', 'run_00_main_pipeline.py'),
    ('01_Preflight', 'run_01_preflight.py'),
    ('02_Pretrain_Representation', 'run_02_pretrain_representation.py'),
    ('03_Evaluate_Representation', 'run_03_evaluate_representation.py'),
    ('04_Pretrain_Few_Shot', 'run_04_pretrain_few_shot.py'),
    ('05_Evaluate_Few_Shot', 'run_05_evaluate_few_shot.py'),
    ('06_CLIP_Baselines', 'run_06_clip_baselines.py'),
    ('07_Pretrain_One_Ablation', 'run_07_pretrain_ablation.py'),
    ('TensorBoard', 'run_tensorboard.py'),
)


def main():
    directory = ROOT / '.run'
    directory.mkdir(exist_ok=True)
    for name, script in ENTRIES:
        path = directory / ('CAME_' + name + '.run.xml')
        if path.exists():
            print('Preserved existing configuration: ' + str(path))
            continue
        component = ET.Element('component', {'name': 'ProjectRunConfigurationManager'})
        config = ET.SubElement(component, 'configuration', {'default': 'false', 'name': 'CAME_' + name,
                               'type': 'PythonConfigurationType', 'factoryName': 'Python', 'singleton': 'true'})
        ET.SubElement(config, 'module', {'name': 'came-repro'})
        options = {'INTERPRETER_OPTIONS': '-u', 'PARENT_ENVS': 'true',
                   'SDK_HOME': str(ROOT / '.venv-came39/Scripts/python.exe').replace('\\', '/'),
                   'WORKING_DIRECTORY': '$PROJECT_DIR$', 'IS_MODULE_SDK': 'false',
                   'ADD_CONTENT_ROOTS': 'true', 'ADD_SOURCE_ROOTS': 'true',
                   'SCRIPT_NAME': '$PROJECT_DIR$/scripts/' + script, 'PARAMETERS': '',
                   'SHOW_COMMAND_LINE': 'true', 'EMULATE_TERMINAL': 'false', 'MODULE_MODE': 'false',
                   'REDIRECT_INPUT': 'false', 'INPUT_FILE': ''}
        for key, value in options.items():
            ET.SubElement(config, 'option', {'name': key, 'value': value})
        envs = ET.SubElement(config, 'envs')
        for key, value in {'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}.items():
            ET.SubElement(envs, 'env', {'name': key, 'value': value})
        ET.SubElement(config, 'method', {'v': '2'})
        ET.indent(component)
        ET.ElementTree(component).write(path, encoding='utf-8', xml_declaration=True)
        print(path)


if __name__ == '__main__':
    main()
