import re

for name in ['figure1_pid_unet_architecture', 'figure2_sgpid_module']:
    html_path = f'paper/{name}.html'
    svg_path = f'paper/{name}.svg'
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    style_m = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
    style = style_m.group(1).strip() if style_m else ''

    svg_m = re.search(r'(<svg.*?</svg>)', content, re.DOTALL)
    if svg_m:
        svg_content = svg_m.group(1)
        if style:
            injected_style = '<style type="text/css"><![CDATA[\n' + style + '\n  ]]></style>\n'
            svg_content = svg_content.replace('<defs>', '<defs>\n    ' + injected_style)

        full_svg = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_content
        with open(svg_path, 'w', encoding='utf-8') as f_out:
            f_out.write(full_svg)
        print(f'Successfully wrote: {svg_path}')
