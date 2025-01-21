# verify_cuda.py

import torch
import logging
import sys

def setup_logging():
    """
    Configure the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def verify_cuda(logger):
    """
    Verify CUDA availability and print GPU details.
    
    Args:
        logger (logging.Logger): The logger instance for logging messages.
    
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info("CUDA is available.")
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of CUDA Devices: {num_gpus}")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert bytes to GB
            current_memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            current_memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            
            logger.info(f"GPU {i}: {gpu_name}")
            logger.info(f"  Total Memory: {total_memory:.2f} GB")
            logger.info(f"  Memory Allocated: {current_memory_allocated:.2f} GB")
            logger.info(f"  Memory Reserved: {current_memory_reserved:.2f} GB")
    else:
        logger.error("CUDA is not available. Please ensure that NVIDIA drivers and CUDA toolkit are properly installed.")
    
    return cuda_available

def main():
    """
    Main function to execute the CUDA verification.
    """
    logger = setup_logging()
    logger.info("Starting CUDA verification...")
    
    cuda_available = verify_cuda(logger)
    
    if cuda_available:
        logger.info("CUDA verification completed successfully.")
    else:
        logger.error("CUDA verification failed. Please refer to the error messages above.")

if __name__ == "__main__":
    main()
