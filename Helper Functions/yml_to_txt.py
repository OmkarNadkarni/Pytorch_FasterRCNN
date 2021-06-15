import ruamel.yaml

yml_path = r'/home/omkarnadkarni/FlangeDetection/environment.yml'
output_path = r'/home/omkarnadkarni/FlangeDetection/requirements.txt'

yaml = ruamel.yaml.YAML()
data = yaml.load(open(yml_path))

requirements = []
for dep in data['dependencies']:
    if isinstance(dep, str):
        package, package_version, python_version = dep.split('=')
        if python_version == '0':
            continue
        requirements.append(package + '==' + package_version)
    elif isinstance(dep, dict):
        for preq in dep.get('pip', []):
            requirements.append(preq)

with open(output_path, 'w') as fp:
    for requirement in requirements:
       print(requirement, file=fp)
