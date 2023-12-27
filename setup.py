from pathlib import Path
import subprocess, sys, re, ssl

def setup_environment():
    # Check Python and PiP version
    verify_python_pip()

    # Install dependencies
    print("Installation des dépendances...\n")
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        print("Dépendances installées avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"Une erreur est survenue lors de l'installation des dépendances: {e}")

    # NLTK config
    setup_nltk()

    # Structure project control
    setup_directories(['data/raw', 'data/processed', 'data/external', 'notebooks', 'src/data_processing', 'src/models', 'src/utils', 'tests'])

def setup_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        
    import nltk
    def download_nltk_data():
        packages = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
        for package in packages:
            nltk.download(package, download_dir=str(Path('env/nltk_data').absolute()))
    
    download_nltk_data()

def verify_python_pip():
    # Check python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 10: 
        raise Exception("Python 3.10 ou une version supérieure est requise.")

    # Check pip version
    pip_version = subprocess.run(["pip", "--version"], capture_output=True, text=True).stdout
    pip_version = re.findall(r'[0-9]{2}\.[0-9]{1,2}\.[0-9]{1,2}', pip_version)[0]
    if int(pip_version[0:2]) < 20:
        raise Exception("Pip 20 ou une version supérieure est requise.")

def setup_directories(directories):
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Dossier '{directory}' créé ou déjà existant.\n")

if __name__ == "__main__":
    try:
        verify_python_pip()
        setup_environment()
        setup_nltk()
    except Exception as e:
        print(e)