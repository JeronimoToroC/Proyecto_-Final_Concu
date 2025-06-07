import os
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.autoinit  # Importante para manejo automático del contexto

# Configuración explícita del entorno
os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
os.environ["PATH"] = f"{os.environ['CUDA_PATH']}\\bin;{os.environ['PATH']}"

try:
    print("🔍 Verificación de PyCUDA con ajustes especiales:")
    print(f"- Dispositivo: {drv.Device(0).name()}")
    print(f"- Capacidad: {drv.Device(0).compute_capability()}")
    print(f"- Versión CUDA: {drv.get_version()}")

    # Kernel modificado con extern "C" para mejor compatibilidad
    kernel_code = """
    extern "C" {
    __global__ void test(float *a) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < 32) {
            a[idx] = idx;
        }
    }
    }
    """

    # Opciones de compilación optimizadas
    mod = SourceModule(
        kernel_code,
        options=[
            '-arch=sm_75',
            '-Xcompiler', '/MD',
            '-Xcompiler', '/wd4819',
            '--use_fast_math',
            '-O3'  # Máxima optimización
        ],
        no_extern_c=False  # Cambiado a False para permitir extern "C"
    )

    # Obtención de la función con nombre decorado
    func = mod.get_function("_Z4testPf")  # Nombre decorado del kernel
    a = np.zeros(32, dtype=np.float32)
    func(drv.Out(a), block=(32,1,1), grid=(1,1))

    print("\n✅ Kernel ejecutado correctamente")
    print(f"Resultado: {a}")

except drv.Error as e:
    print(f"\n❌ Error de CUDA: {str(e)}")
    print("\nSolución definitiva:")
    print("1. Abre 'x64 Native Tools Command Prompt for VS 2022'")
    print("2. Ejecuta: pip install --force-reinstall pycuda==2024.1.2")
    print("3. Vuelve a ejecutar este script en esa misma ventana")

except Exception as e:
    print(f"\n❌ Error general: {str(e)}")