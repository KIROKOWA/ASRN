# Copyright 2026 Alexis Calvario
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""python -m asrn ...: entrenar un modelo ASRN con la capa estable (la linea de comandos de asrn.nucleo; ver
python -m asrn --help)."""
from . import estable, nucleo


def main(args=None):
    """Entrenar desde la linea de comandos con la capa estable (tambien el comando asrn-train de pyproject.toml)."""
    return nucleo._principal(args, preparar=estable.convertir)


if __name__ == '__main__':
    import sys
    sys.argv[0] = 'python -m asrn'      # el nombre que muestra --help (si no, '__main__.py'); asrn-train no llega aqui
    main()
