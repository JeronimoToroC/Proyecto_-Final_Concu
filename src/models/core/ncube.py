from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context
import os

os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8"
os.environ["PATH"] = f"{os.environ['CUDA_PATH']}\\bin;{os.environ['PATH']}"

try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    from pycuda.tools import make_default_context
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"[CUDA] Advertencia: PyCUDA no disponible - {str(e)}")
    CUDA_AVAILABLE = False

@dataclass(frozen=True)
class NCube:
    """
    N-cubo hace referencia a un cubo n-dimensional, donde estarán indexados según la posición de precedencia de los datos, permitiendo el rápido acceso y operación en memoria.
    - `indice`: índice original del n-cubo asociado con un literal (0:A, 1:B, 2:C, ...) que permita representabilidad en su alcance o tiempo futuro.
    - `dims`: dimensiones activas actuales del n-cubo, es aquí donde se conoce la dimensionalidad según su cantidad de elementos, de forma tal que si este en el tiempo es condicionado o marginalizado tendrá una dimensionalidad menor o igual a la original a pesar que haya una alta dimensión específica.
    - `data`: arreglo numpy con los datos indexados según la notación de origen, de ser necesario se aplica una transformación sobre estos que los reindexe si se desea otra notación particular.
    """

    indice: int
    dims: NDArray[np.int8]
    data: np.ndarray

    # Configuración CUDA
    _CUDA_INITIALIZED = False
    _CUDA_PATH = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
    _context = None
    _mod = None
    _marginalize_kernel = None
    
    _MARGINALIZE_KERNEL_CODE = """
    __global__ void marginalize_kernel(
        const float *input,
        float *output,
        const int input_size,
        const int output_size,
        const int reduce_stride
    ) {
        int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (out_idx >= output_size) return;
        
        float sum = 0.0f;
        int count = 0;
        
        for(int i = out_idx * reduce_stride; i < (out_idx+1) * reduce_stride; i++) {
            if(i < input_size) {
                sum += input[i];
                count++;
            }
        }
        
        if(count > 0) {
            output[out_idx] = sum / count;
        }
    }
    """
    
    @classmethod
    def _initialize_cuda(cls):
        """Inicialización segura del entorno CUDA"""
        if cls._CUDA_INITIALIZED:
            return
            
        try:
            # Configurar entorno
            os.environ["CUDA_PATH"] = cls._CUDA_PATH
            os.environ["PATH"] = f"{cls._CUDA_PATH}\\bin;{os.environ['PATH']}"
            
            # Verificar dispositivo
            device = drv.Device(0)
            print(f"[CUDA] Configurando para dispositivo: {device.name()}")
            
            cls._context = make_default_context()
            
            # Compilar kernel
            cls._mod = SourceModule(
                cls._MARGINALIZE_KERNEL_CODE,
                options=['-arch=sm_75', '--ptxas-options=-v'],
                no_extern_c=True
            )
            cls._marginalize_kernel = cls._mod.get_function("marginalize_kernel")
            
            cls._CUDA_INITIALIZED = True
            
        except Exception as e:
            print(f"[CUDA] Error de inicialización: {str(e)}")
            cls._CUDA_INITIALIZED = False

    def __post_init__(self):
        """Validación de datos + inicialización CUDA"""
        if self.dims.size and self.data.shape != (2,) * self.dims.size:
            raise ValueError(
                f"Forma inválida {self.data.shape} para dimensiones {self.dims}"
            )
        
        # Asegurar tipo float32 para CUDA
        object.__setattr__(self, 'data', self.data.astype(np.float32))
        
        # Inicializar CUDA silenciosamente
        self._initialize_cuda()

    def condicionar(
        self,
        indices_condicionados: NDArray[np.int8],
        estado_inicial: NDArray[np.int8],
    ) -> "NCube":
        """
        Aplicar condiciones de fondo sobre un n-cubo. En estas lo que se hace es seleccionar una serie de caras sobre el n-cubo según las dimensiones escogidas y su estado inicial específico asociado, descartandose así todas las demás que no pertenezcan al indice condicionado.
        En la selección de las dimensiones es importante saber cómo la dimensión más externa es la más significativa, de forma que la selección debe hacerse de afuera hacia adentro.
        Debe tenerse claro también la localidad de las dimensiones puesto aunque se tengan dimensiones muy superiores no hay correspondencia con el total de dimensiones del cubo (dimensiones locales).

        Args:
        ----------
            indices_condicionados (NDArray[np.int8]): Dimensiones o ejes en los cuales se aplicará el condicinamiento.
            estado_inicial (NDArray[np.int8]): El estado inicial asociado al sistema.

        Returns:
        -------

            NCube: El n-cubo seleccionado en todos los ejes, pero se definen para dar selección los cuales se hayan enviado como parámetros.

        Example:
        -------
        El n-cubo original está asociado con el estado inicial

        >>> estado_inicial = np.array([1,0,0])
        >>> mi_ncubo
            NCube(index=(1,)):
                dims=(0, 1, 2)
                shape=(2, 2, 2)
                data=
                    [[[0.1  0.3 ]
                    [0.5  0.7 ]]
                    [[0.9  0.11]
                    [0.13 0.15]]]
        >>> dimensiones = np.array([2])
        >>> mi_ncubo.condicionar(dimensiones, estado_incial)
            NCube(index=(1,)):
                dims=(0, 1)
                shape=(2, 2)
                data=
                    [[0.1 0.3]
                    [0.5 0.7]]
        """
        numero_dims = self.dims.size
        seleccion = [slice(None)] * numero_dims

        for condicion in indices_condicionados:
            level_arr = numero_dims - (condicion + 1)
            seleccion[level_arr] = estado_inicial[condicion]

        nuevas_dims = np.array(
            [dim for dim in self.dims if dim not in indices_condicionados],
            dtype=np.int8,
        )
        return NCube(
            data=self.data[tuple(seleccion)],
            dims=nuevas_dims,
            indice=self.indice,
        )
    indice: int
    dims: NDArray[np.int8]
    data: np.ndarray

    # Configuración CUDA
    _CUDA_INITIALIZED = False
    _CUDA_PATH = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
    
    if CUDA_AVAILABLE:
        _MARGINALIZE_KERNEL_CODE = """
        __global__ void marginalize_kernel(
            const float *input,
            float *output,
            const int input_size,
            const int output_size,
            const int reduce_stride
        ) {
            int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (out_idx >= output_size) return;
            
            float sum = 0.0f;
            int count = 0;
            
            for(int i = out_idx * reduce_stride; i < (out_idx+1) * reduce_stride; i++) {
                if(i < input_size) {
                    sum += input[i];
                    count++;
                }
            }
            
            if(count > 0) {
                output[out_idx] = sum / count;
            }
        }
        """
    else:
        _MARGINALIZE_KERNEL_CODE = None

    @classmethod
    def _initialize_cuda(cls):
        """Inicialización segura del entorno CUDA"""
        if not CUDA_AVAILABLE or cls._CUDA_INITIALIZED:
            return
            
        try:
            # Configurar entorno
            os.environ["CUDA_PATH"] = cls._CUDA_PATH
            os.environ["PATH"] = f"{cls._CUDA_PATH}\\bin;{os.environ['PATH']}"
            
            # Verificar dispositivo
            device = drv.Device(0)
            print(f"[CUDA] Configurando para dispositivo: {device.name()}")
            
            # Crear contexto
            cls._context = make_default_context()
            
            # Compilar kernel
            cls._mod = SourceModule(
                cls._MARGINALIZE_KERNEL_CODE,
                options=['-arch=sm_75'],
                no_extern_c=True
            )
            cls._marginalize_kernel = cls._mod.get_function("marginalize_kernel")
            
            cls._CUDA_INITIALIZED = True
            
        except Exception as e:
            print(f"[CUDA] Error de inicialización: {str(e)}")
            cls._CUDA_INITIALIZED = False
            if hasattr(cls, '_context') and cls._context:
                cls._context.pop()

    def __post_init__(self):
        """Validación de datos"""
        if self.dims.size and self.data.shape != (2,) * self.dims.size:
            raise ValueError(
                f"Forma inválida {self.data.shape} para dimensiones {self.dims}"
            )
        
        # Asegurar tipo float32
        object.__setattr__(self, 'data', self.data.astype(np.float32))
        
        # Inicializar CUDA solo si está disponible
        if CUDA_AVAILABLE:
            self._initialize_cuda()

    def marginalizar(self, ejes: NDArray[np.int8]) -> "NCube":
        """
        Versión híbrida que usa GPU si está disponible, si no usa CPU
        """
        # Validación común
        marginable_axis = np.intersect1d(ejes, self.dims)
        if not marginable_axis.size:
            return self
            
        # Cálculo de dimensiones
        numero_dims = self.dims.size - 1
        ejes_locales = tuple(
            numero_dims - dim_idx
            for dim_idx, axis in enumerate(self.dims)
            if axis in marginable_axis
        )
        new_dims = np.array(
            [d for d in self.dims if d not in marginable_axis],
            dtype=np.int8,
        )
        
        # Intentar GPU solo si está disponible e inicializada
        if CUDA_AVAILABLE and self._CUDA_INITIALIZED:
            try:
                input_data = self.data
                input_size = input_data.size
                output_shape = [s for i, s in enumerate(self.data.shape) 
                              if i not in ejes_locales]
                output_size = np.prod(output_shape)
                reduce_stride = input_size // output_size
                
                # Configurar ejecución
                block_size = 256
                grid_size = (output_size + block_size - 1) // block_size
                
                # Reservar memoria
                input_gpu = drv.mem_alloc(input_data.nbytes)
                output_gpu = drv.mem_alloc(output_size * np.dtype(np.float32).itemsize)
                drv.memcpy_htod(input_gpu, input_data)
                
                # Ejecutar kernel
                self._marginalize_kernel(
                    input_gpu, output_gpu,
                    np.int32(input_size), np.int32(output_size), np.int32(reduce_stride),
                    block=(block_size, 1, 1), grid=(grid_size, 1)
                )
                
                # Obtener resultados
                output_data = np.empty(output_shape, dtype=np.float32)
                drv.memcpy_dtoh(output_data, output_gpu)
                
                return NCube(
                    data=output_data,
                    dims=new_dims,
                    indice=self.indice
                )
                
            except Exception as e:
                print(f"[CUDA] Error en GPU, usando CPU: {str(e)}")
        
        # Versión CPU
        return NCube(
            data=np.mean(self.data, axis=ejes_locales, keepdims=False),
            dims=new_dims,
            indice=self.indice,
        )

    def __del__(self):
            """Limpieza del contexto CUDA"""
            if self._CUDA_INITIALIZED and self._context:
                self._context.pop()
            
    def __str__(self) -> str:
        dims_str = f"dims={self.dims}"
        forma_str = f"shape={self.data.shape}"
        datos_str = str(self.data).replace("\n", "\n" + " " * 8)
        return (
            f"NCube(index={self.indice}):\n"
            f"    {dims_str}\n"
            f"    {forma_str}\n"
            f"    data=\n        {datos_str}"
        )
