# Vision Prompt Composer

Custom node para ComfyUI que gera um único texto usando várias referências visuais. Reutiliza a geração e o schema de amostragem do `Generate Text` nativo, substituindo a entrada `image` por oito entradas opcionais, `image_1` a `image_8`.

## Uso

1. Conecte o `CLIPLoader` a `clip` e seu texto a `prompt`.
2. Conecte cada `Load Image` a uma entrada de imagem.
3. Conecte `generated_text` ao preview ou ao consumidor do prompt.
4. Execute o workflow. O modelo recebe todas as imagens em uma única chamada.

Não precisa conectar todas as entradas. A numeração `<Picture N>` segue a ordem das entradas conectadas, pulando as vazias. Por exemplo, `image_1` e `image_3` tornam-se `<Picture 1>` e `<Picture 2>`. Em batches, cada item recebe seu próprio número antes de passar à próxima entrada. As imagens mantêm suas dimensões até o pré-processamento nativo do modelo; não são concatenadas, redimensionadas pelo node nem transformadas em colagem.

Exemplo de pedido:

```text
Mode: Ref2VA
Duration: 8 seconds
Use <Picture 1> as the character reference and <Picture 2> as the environment reference.
Describe both references and place the character in that environment.
Camera: medium shot, static camera.
Audio: quiet ambience, no music, no dialogue.
```

Os papéis das referências devem ser informados no pedido. Conectar duas imagens não determina automaticamente primeiro e último frames. Para FL2VA, declare esses papéis explicitamente.

## Compatibilidade

- Desenvolvido para o CLIP visual Qwen com suporte a `images=[...]`; validado no tokenizer Qwen usado pelo ambiente ComfyUI 0.34.5.
- Preserva `max_length`, sampling on/off, temperatura, top-k/p, min-p, penalidades, seed, thinking e template do node nativo.
- Sem imagens, delega diretamente ao node nativo.
- Os sockets de vídeo e áudio do original são preservados; suporte efetivo depende do modelo. A validação multimodal deste projeto cobre imagens.
- Verifica quantidade e ordem dos payloads visuais antes da geração. Um modelo/template que ignore imagens produz um erro claro em vez de uma descrição sem acesso às referências.
- Templates manuais precisam conter um placeholder visual nativo por imagem. O template padrão é recomendado.
- A capacidade total de referências depende da memória e do contexto do modelo. Oito sockets não impõem um limite de oito frames em batches.

## Instalação e testes

Na pasta `ComfyUI/custom_nodes`, execute:

```sh
git clone https://github.com/gabxav/ComfyUI-Vision-Prompt-Composer.git
```

Reinicie o ComfyUI com a fila vazia e atualize o navegador. Procure **Vision Prompt Composer** na categoria `text`. Não há dependências adicionais às do ComfyUI.

Se já instalou a versão anterior em `ComfyUI-MultiImage-Text`, substitua essa pasta pela nova instalação; não mantenha as duas cópias. O identificador interno `TextGenerateMultiImage` foi preservado para compatibilidade com os workflows existentes.

Na pasta raiz do ComfyUI, usando seu ambiente Python:

```sh
python custom_nodes/ComfyUI-Vision-Prompt-Composer/test_nodes.py --cpu
```

Os testes usam o tokenizer real sem carregar pesos do modelo. Cobrem sockets vazios, resoluções distintas, todos os oito sockets, batches, perda de imagens em templates manuais, passagem dos controles de sampling, texto sem imagens e tensors inválidos.

## Fontes

- [Generate Text nativo](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_textgen.py)
- [Tokenizer visual Qwen](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/qwen35.py)
