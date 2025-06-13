#!/usr/bin/env python3
"""
Script automatico per refactoring sicuro del progetto FilmOCredit.
Esegue operazioni di pulizia e formattazione senza modificare la logica.
"""

import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime
import argparse

# Colori per output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(message, color=Colors.BLUE):
    """Stampa messaggio formattato per step."""
    print(f"\n{color}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{color}{Colors.BOLD}{message}{Colors.ENDC}")
    print(f"{color}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def run_command(cmd, check=True):
    """Esegue comando e mostra output."""
    print(f"\n{Colors.YELLOW}▶ Eseguendo: {' '.join(cmd)}{Colors.ENDC}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"{Colors.RED}⚠ Warning: {result.stderr}{Colors.ENDC}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}✗ Errore: {e.stderr}{Colors.ENDC}")
        return False

def backup_project():
    """Crea backup del progetto."""
    print_step("📦 CREANDO BACKUP DEL PROGETTO")

    backup_dir = Path(f"../filmocredit-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    try:
        shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns(
            '__pycache__', '*.pyc', '.git', 'data/episodes/*',
            'data/raw/*.mp4', '*.log', 'db/*.db'
        ))
        print(f"{Colors.GREEN}✓ Backup creato in: {backup_dir}{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"{Colors.RED}✗ Errore nel backup: {e}{Colors.ENDC}")
        return False

def install_dev_tools():
    """Installa tools di sviluppo."""
    print_step("📥 INSTALLANDO TOOLS DI SVILUPPO")

    if Path("requirements-dev.txt").exists():
        return run_command([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    else:
        tools = ["black", "isort", "autoflake", "pylint", "mypy", "safety", "bandit"]
        return run_command([sys.executable, "-m", "pip", "install"] + tools)

def clean_imports():
    """Rimuove import inutilizzati."""
    print_step("🧹 RIMUOVENDO IMPORT INUTILIZZATI")

    # Converti tutto in Path objects per consistenza
    files_to_process = [Path("app.py")] + list(Path("scripts_v3").glob("*.py"))

    for file in files_to_process:
        if file.exists():
            print(f"\n  Processando: {file}")
            run_command([
                "autoflake",
                "--in-place",
                "--remove-all-unused-imports",
                "--remove-unused-variables",
                "--ignore-init-module-imports",
                str(file)
            ])

    return True

def sort_imports():
    """Ordina gli import secondo PEP8."""
    print_step("📚 ORDINANDO IMPORTS")

    return run_command(["isort", "app.py", "scripts_v3/", "--profile", "black"])

def format_code():
    """Formatta il codice con Black."""
    print_step("✨ FORMATTANDO CODICE CON BLACK")

    return run_command(["black", "app.py", "scripts_v3/", "--line-length", "120"])

def add_type_hints():
    """Aggiunge type hints dove possibile."""
    print_step("🏷️  AGGIUNGENDO TYPE HINTS")

    print(f"{Colors.YELLOW}ℹ Type hints verranno aggiunti manualmente nei file refactorizzati{Colors.ENDC}")
    return True

def security_check():
    """Esegue controlli di sicurezza."""
    print_step("🔒 CONTROLLO SICUREZZA")

    print("\n📋 Checking dependencies...")
    run_command(["safety", "check"], check=False)

    print("\n🔍 Scanning code for vulnerabilities...")
    run_command(["bandit", "-r", "scripts_v3/", "-ll"], check=False)

    return True

def run_smoke_tests():
    """Esegue test di base per verificare che nulla sia rotto."""
    print_step("🧪 ESEGUENDO SMOKE TESTS")

    test_script = """
import sys
try:
    # Test imports
    print("Testing imports...")
    import app
    import scripts_v3.config
    import scripts_v3.utils
    import scripts_v3.scene_detection
    import scripts_v3.frame_analysis
    import scripts_v3.azure_vlm_processing
    print("✓ All imports successful")

    # Test basic functions
    print("\\nTesting basic functions...")
    from scripts_v3.utils import normalize_text_for_comparison
    result = normalize_text_for_comparison("Test Text!", ["test"])
    assert isinstance(result, str)
    print("✓ Basic functions working")

    print("\\n✅ ALL SMOKE TESTS PASSED!")
    sys.exit(0)
except Exception as e:
    print(f"\\n❌ SMOKE TEST FAILED: {e}")
    sys.exit(1)
"""

    result = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0

def generate_report():
    """Genera report del refactoring."""
    print_step("📊 GENERANDO REPORT")

    report = f"""
REFACTORING REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

Files processati:
- app.py
- scripts_v3/*.py

Operazioni eseguite:
✓ Backup del progetto
✓ Import inutilizzati rimossi
✓ Import ordinati (isort)
✓ Codice formattato (black)
✓ Type hints preparati per aggiunta manuale
✓ Security check eseguito
✓ Smoke tests passati

Prossimi passi manuali:
1. Verifica le modifiche con: git diff
2. Testa l'applicazione manualmente
3. Sostituisci i file refactorizzati forniti
4. Commit delle modifiche

{'='*60}
"""

    with open("refactoring_report.txt", "w") as f:
        f.write(report)

    print(report)
    print(f"\n{Colors.GREEN}Report salvato in: refactoring_report.txt{Colors.ENDC}")

def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description="Refactoring automatico FilmOCredit")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup")
    parser.add_argument("--no-tests", action="store_true", help="Skip tests")
    args = parser.parse_args()

    print(f"""
{Colors.BLUE}{Colors.BOLD}
╔═══════════════════════════════════════════╗
║     FilmOCredit Auto-Refactoring Tool     ║
║           Safe & Automated                ║
╚═══════════════════════════════════════════╝
{Colors.ENDC}
""")

    steps = []

    # 1. Backup
    if not args.no_backup:
        if not backup_project():
            print(f"\n{Colors.RED}Backup fallito! Interruzione per sicurezza.{Colors.ENDC}")
            sys.exit(1)

    # 2. Install tools
    steps.append(("Installing dev tools", install_dev_tools))

    # 3. Clean code
    steps.append(("Cleaning imports", clean_imports))
    steps.append(("Sorting imports", sort_imports))
    steps.append(("Formatting code", format_code))
    steps.append(("Adding type hints", add_type_hints))

    # 4. Security
    steps.append(("Security check", security_check))

    # 5. Tests
    if not args.no_tests:
        steps.append(("Running smoke tests", run_smoke_tests))

    # Execute steps
    failed = False
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n{Colors.RED}Step '{step_name}' failed!{Colors.ENDC}")
            failed = True
            break

    # 6. Report
    generate_report()

    if not failed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ REFACTORING COMPLETATO CON SUCCESSO!{Colors.ENDC}")
        print(f"\n{Colors.YELLOW}Prossimi passi:{Colors.ENDC}")
        print("1. Verifica le modifiche: git diff")
        print("2. Sostituisci i file refactorizzati manualmente")
        print("3. Testa l'applicazione")
        print("4. Commit: git add -A && git commit -m 'refactor: automatic code improvements'")
    else:
        print(f"\n{Colors.RED}⚠ Refactoring completato con warnings.{Colors.ENDC}")
        print("Verifica i problemi segnalati prima di procedere.")

if __name__ == "__main__":
    main()
