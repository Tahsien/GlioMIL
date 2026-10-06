import os
import subprocess
import urllib.request
import json

try:
    import radiomics
    print(f"Already installed: {radiomics.__version__}")
except ImportError:
    with urllib.request.urlopen('https://pypi.org/pypi/pyradiomics/3.0.1/json') as r:
        pypi_data = json.load(r)
    tar_url = next(f['url'] for f in pypi_data['urls'] if f['filename'].endswith('.tar.gz'))

    build_dir = '/kaggle/working/pyrad_build'
    os.makedirs(build_dir, exist_ok=True)
    orig_cwd = os.getcwd()
    os.chdir(build_dir)

    subprocess.run(['curl', '-sL', '-o', 'pyradiomics.tar.gz', tar_url], check=True)
    subprocess.run(['tar', 'xzf', 'pyradiomics.tar.gz'], check=True)
    src_dir = [d for d in os.listdir('.') if d.startswith('pyradiomics-')][0]
    versioneer_path = os.path.join(src_dir, 'versioneer.py')

    with open(versioneer_path) as f:
        content = f.read()
    content = content.replace('configparser.SafeConfigParser()', 'configparser.ConfigParser()')
    content = content.replace('parser.readfp(f)', 'parser.read_file(f)')
    with open(versioneer_path, 'w') as f:
        f.write(content)

    subprocess.run(['pip', 'install', 'SimpleITK', 'PyWavelets', 'pykwalify', 'six'], check=True)
    result = subprocess.run(['pip', 'install', '.', '--no-build-isolation'],
                             cwd=src_dir, capture_output=True, text=True)
    print(result.stdout[-1000:])
    if result.returncode != 0:
        print("INSTALL FAILED:")
        print(result.stderr[-1500:])
    os.chdir(orig_cwd)

    import radiomics
    print(f"SUCCESS: pyradiomics {radiomics.__version__} installed")