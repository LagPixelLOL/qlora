from qlora import train

if __name__ == "__main__":
    from flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    train()