"""
Model Architecture Diagrams

Visualizations of the four parallelism classification models.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from qhchina.helpers import load_fonts

load_fonts(target_font='Noto Sans CJK TC')

# Output directory (same folder as this script)
FIGURES_DIR = Path(__file__).parent

# Unified color palette (blues and grays)
COLORS = {
    'special_token': '#1a365d',
    'text_token': '#4a7c9b',
    'bert': '#2c5282',
    'classifier': '#3182ce',
    'output': '#63b3ed',
    'arrow': '#4a5568',
    'embedding': '#bee3f8',
}

FONT_EN = 'Avenir'
FONT_ZH = 'Noto Sans CJK TC'

# Font sizes (unified across all diagrams)
TOKEN_FONT_SIZE = 13
SPECIAL_TOKEN_FONT_SIZE = 12
WIDE_FONT_SIZE = 14
OUTPUT_FONT_SIZE = 13
LABEL_FONT_SIZE = 10

# Component dimensions (unified across all diagrams)
TOKEN_BOX_H = 0.035           # Height of token boxes
TOKEN_BOX_W_ZH = 0.028        # Width for Chinese character tokens
TOKEN_BOX_W_SPECIAL = 0.038   # Width for special tokens like [CLS], [SEP], [CP1]
BERT_H = 0.065                # Height of BERT encoder box
EMBED_RADIUS = 0.02           # Radius of embedding circles
CLF_H = 0.055                 # Height of classifier box
OUTPUT_H = 0.045              # Height of output box
OUTPUT_W = 0.10               # Width of output box

# Arrow gap between components (consistent across all diagrams)
ARROW_GAP = 0.08
ARROW_PAD = 0.02            # Padding at arrow endpoints

# Punctuation text offset
PUNCT_TEXT_OFFSET = 0.000 # some fonts need it


def draw_token_box(ax, x, y, text, color, width=0.04, height=0.04, fontsize=11, is_chinese=False, text_offset_x=0):
    """Draw a token box centered at (x, y). text_offset_x shifts the text within the box."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle='round,pad=0.01,rounding_size=0.01',
                          facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    font = FONT_ZH if is_chinese else FONT_EN
    ax.text(x + text_offset_x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', fontfamily=font)


def draw_wide_box(ax, x, y, text, color, width=0.4, height=0.065):
    """Draw a wider box for BERT/classifier"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle='round,pad=0.01,rounding_size=0.02',
                          facecolor=color, edgecolor='white', linewidth=2, alpha=0.95)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=WIDE_FONT_SIZE,
            fontweight='bold', color='white', fontfamily=FONT_EN)
    return y - height/2, y + height/2


def draw_arrow(ax, start, end):
    """Draw an arrow from start to end"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5))


def draw_embedding_circle(ax, x, y, label=None, radius=0.02, label_position='right'):
    """
    Draw a circle for embedding.
    label_position: 'right' (default), 'top', 'left', 'bottom', 'middle'
    """
    circle = plt.Circle((x, y), radius, facecolor=COLORS['embedding'],
                         edgecolor=COLORS['bert'], linewidth=2)
    ax.add_patch(circle)
    if label:
        if label_position == 'right':
            ax.text(x + radius + 0.02, y, label, ha='left', va='center',
                    fontsize=10, fontfamily=FONT_EN, style='italic', color='#4a5568')
        elif label_position == 'top':
            ax.text(x, y + radius + 0.015, label, ha='center', va='bottom',
                    fontsize=10, fontfamily=FONT_EN, style='italic', color='#4a5568')
        elif label_position == 'left':
            ax.text(x - radius - 0.02, y, label, ha='right', va='center',
                    fontsize=10, fontfamily=FONT_EN, style='italic', color='#4a5568')
        elif label_position == 'bottom':
            ax.text(x, y - radius - 0.015, label, ha='center', va='top',
                    fontsize=10, fontfamily=FONT_EN, style='italic', color='#4a5568')
        elif label_position == 'middle':
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=10, fontfamily=FONT_EN, style='italic', color='#4a5568')


# =============================================================================
# 1. Character-Pair Model
# =============================================================================

def draw_char_model():
    # Token spacing for this diagram
    spacing = 0.09
    padding = 0.04

    tokens = [('[CLS]', COLORS['special_token'], False),
              ('避', COLORS['text_token'], True),
              ('[SEP]', COLORS['special_token'], False),
              ('安', COLORS['text_token'], True),
              ('[SEP]', COLORS['special_token'], False)]

    # Calculate positions first to determine canvas size
    n_tokens = len(tokens)
    total_token_width = (n_tokens - 1) * spacing
    
    # Vertical positions (from bottom up)
    y_tokens = padding
    tokens_top = y_tokens + TOKEN_BOX_H / 2
    
    bert_bottom = tokens_top + ARROW_GAP
    y_bert = bert_bottom + BERT_H / 2
    bert_top = y_bert + BERT_H / 2
    
    embed_bottom = bert_top + ARROW_GAP
    y_embed = embed_bottom + EMBED_RADIUS
    embed_top = y_embed + EMBED_RADIUS
    
    clf_bottom = embed_top + ARROW_GAP
    y_classifier = clf_bottom + CLF_H / 2
    clf_top = y_classifier + CLF_H / 2
    
    output_bottom = clf_top + ARROW_GAP
    y_output = output_bottom + OUTPUT_H / 2
    output_top = y_output + OUTPUT_H / 2

    # Calculate canvas bounds
    y_min = 0
    y_max = output_top + padding
    
    # Horizontal: based on token positions (centered at 0.5)
    start_x = 0.5 - total_token_width / 2
    end_x = 0.5 + total_token_width / 2
    x_min = start_x - TOKEN_BOX_W_SPECIAL / 2 - padding
    x_max = end_x + TOKEN_BOX_W_SPECIAL / 2 + padding
    
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    
    # Create figure with proper aspect ratio
    fig_scale = 10
    fig, ax = plt.subplots(1, 1, figsize=(canvas_width * fig_scale, canvas_height * fig_scale))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')
    ax.set_aspect('equal')

    token_positions = []
    for i, (text, color, is_zh) in enumerate(tokens):
        x = start_x + i * spacing
        w = TOKEN_BOX_W_ZH if is_zh else TOKEN_BOX_W_SPECIAL
        draw_token_box(ax, x, y_tokens, text, color, width=w, height=TOKEN_BOX_H, 
                      fontsize=TOKEN_FONT_SIZE, is_chinese=is_zh)
        token_positions.append(x)

    draw_wide_box(ax, 0.5, y_bert, 'BERT Encoder', COLORS['bert'], height=BERT_H)

    for x in token_positions:
        draw_arrow(ax, (x, tokens_top + ARROW_PAD), (x, bert_bottom - ARROW_PAD))

    draw_embedding_circle(ax, 0.5, y_embed, '[CLS] embedding', radius=EMBED_RADIUS)
    draw_arrow(ax, (0.5, bert_top + ARROW_PAD), (0.5, embed_bottom - ARROW_PAD))

    draw_wide_box(ax, 0.5, y_classifier, 'Linear Classifier', COLORS['classifier'], width=0.32, height=CLF_H)
    draw_arrow(ax, (0.5, embed_top + ARROW_PAD), (0.5, clf_bottom - ARROW_PAD))

    draw_token_box(ax, 0.5, y_output, 'P / not P', COLORS['output'], width=OUTPUT_W, height=OUTPUT_H, fontsize=OUTPUT_FONT_SIZE)
    draw_arrow(ax, (0.5, clf_top + ARROW_PAD), (0.5, output_bottom - ARROW_PAD))

    ax.text(0.5 + OUTPUT_W/2 + 0.02, y_output, 'Parallel / Not Parallel', ha='left', va='center', 
            fontsize=LABEL_FONT_SIZE, fontfamily=FONT_EN, style='italic', color='#718096')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'arch_char_model.png', bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# 2. Couplet Model
# =============================================================================

def draw_coup_model():
    # Token spacing for this diagram
    spacing = 0.055
    padding = 0.04
    special_offset = 0.012

    # Example couplet
    line1 = '避俗嫌林淺'
    line2 = '安貧覺屋寬'
    

    tokens = [('[CLS]', COLORS['special_token'], False)]
    for ch in line1:
        tokens.append((ch, COLORS['text_token'], True))
    tokens.append(('，', COLORS['text_token'], True))
    for ch in line2:
        tokens.append((ch, COLORS['text_token'], True))
    tokens.append(('[SEP]', COLORS['special_token'], False))

    n_tokens = len(tokens)
    total_token_width = (n_tokens - 1) * spacing + 2 * special_offset  # Account for CLS/SEP offsets

    # Vertical positions (from bottom up)
    y_tokens = padding
    tokens_top = y_tokens + TOKEN_BOX_H / 2
    
    bert_bottom = tokens_top + ARROW_GAP
    y_bert = bert_bottom + BERT_H / 2
    bert_top = y_bert + BERT_H / 2
    
    embed_bottom = bert_top + ARROW_GAP
    y_embed = embed_bottom + EMBED_RADIUS
    embed_top = y_embed + EMBED_RADIUS
    
    clf_bottom = embed_top + ARROW_GAP
    y_classifier = clf_bottom + CLF_H / 2
    clf_top = y_classifier + CLF_H / 2
    
    output_bottom = clf_top + ARROW_GAP
    y_output = output_bottom + OUTPUT_H / 2
    output_top = y_output + OUTPUT_H / 2

    # Calculate canvas bounds
    y_min = 0
    y_max = output_top + padding
    
    # Use BERT width (0.75) as reference since it's the widest element
    x_content_width = max(total_token_width + TOKEN_BOX_W_SPECIAL, 0.75)
    x_min = 0.5 - x_content_width / 2 - padding
    x_max = 0.5 + x_content_width / 2 + padding
    
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    
    # Create figure with proper aspect ratio
    fig_scale = 10
    fig, ax = plt.subplots(1, 1, figsize=(canvas_width * fig_scale, canvas_height * fig_scale))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')
    ax.set_aspect('equal')

    total_width = (n_tokens - 1) * spacing
    start_x = 0.5 - total_width / 2

    token_positions = []
    for i, (text, color, is_zh) in enumerate(tokens):
        x = start_x + i * spacing
        # Push [CLS] further left, [SEP] further right
        if text == '[CLS]':
            x -= special_offset
        elif text == '[SEP]':
            x += special_offset
        w = TOKEN_BOX_W_ZH if is_zh else TOKEN_BOX_W_SPECIAL
        text_offset = PUNCT_TEXT_OFFSET if text in ('，', '。') else 0
        draw_token_box(ax, x, y_tokens, text, color, width=w, height=TOKEN_BOX_H,
                      fontsize=TOKEN_FONT_SIZE, is_chinese=is_zh, text_offset_x=text_offset)
        token_positions.append(x)

    draw_wide_box(ax, 0.5, y_bert, 'BERT Encoder', COLORS['bert'], width=0.75, height=BERT_H)

    for x in token_positions:
        draw_arrow(ax, (x, tokens_top + ARROW_PAD), (x, bert_bottom - ARROW_PAD))

    draw_embedding_circle(ax, 0.5, y_embed, '[CLS] embedding', radius=EMBED_RADIUS)
    draw_arrow(ax, (0.5, bert_top + ARROW_PAD), (0.5, embed_bottom - ARROW_PAD))

    draw_wide_box(ax, 0.5, y_classifier, 'Linear Classifier', COLORS['classifier'], width=0.32, height=CLF_H)
    draw_arrow(ax, (0.5, embed_top + ARROW_PAD), (0.5, clf_bottom - ARROW_PAD))

    draw_token_box(ax, 0.5, y_output, 'P / not P', COLORS['output'], width=OUTPUT_W, height=OUTPUT_H, fontsize=OUTPUT_FONT_SIZE)
    draw_arrow(ax, (0.5, clf_top + ARROW_PAD), (0.5, output_bottom - ARROW_PAD))

    ax.text(0.5 + OUTPUT_W/2 + 0.02, y_output, 'Parallel / Not Parallel', ha='left', va='center',
            fontsize=LABEL_FONT_SIZE, fontfamily=FONT_EN, style='italic', color='#718096')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'arch_coup_model.png', bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# 3. Poem-4 Model (Full Poem, 4 Outputs)
# =============================================================================

def draw_poem4_model():
    # Example poem (4 couplets)
    couplets = [
        ('感別情偏切', '離筵酒易酣'),
        ('孤舟搖霽月', '一騎入晴嵐'),
        ('戀闕心懸北', '思親夢繞南'),
        ('明朝倚樓處', '雲樹思難堪'),
    ]

    # Token spacing for this diagram
    spacing = 0.048
    line_spacing = 0.06
    padding = 0.04
    special_offset = 0.008

    # Calculate column positions for aligned tokens
    n_columns = 13  # [CP], char1-5, comma, char6-10, period
    total_token_width = (n_columns - 1) * spacing

    # Vertical positions (from bottom up)
    y_start = padding
    bottom_row_y = y_start
    top_row_y = y_start + 3 * line_spacing
    tokens_top = top_row_y + TOKEN_BOX_H / 2
    
    bert_bottom = tokens_top + ARROW_GAP
    y_bert = bert_bottom + BERT_H / 2
    bert_top = y_bert + BERT_H / 2
    
    embed_bottom = bert_top + ARROW_GAP
    y_embed = embed_bottom + EMBED_RADIUS
    embed_top = y_embed + EMBED_RADIUS
    
    clf_bottom = embed_top + ARROW_GAP
    y_classifier = clf_bottom + CLF_H / 2
    clf_top = y_classifier + CLF_H / 2
    
    output_bottom = clf_top + ARROW_GAP
    y_output = output_bottom + OUTPUT_H / 2
    output_top = y_output + OUTPUT_H / 2

    # Calculate canvas bounds
    y_min = 0
    y_max = output_top + padding
    
    # Horizontal: include CLS on left, SEP on right
    start_x = 0.5 - total_token_width / 2
    column_x = [start_x + i * spacing for i in range(n_columns)]
    cls_x = column_x[0] - spacing - special_offset * 2.5
    sep_x = column_x[12] + spacing + special_offset
    
    x_min = cls_x - TOKEN_BOX_W_SPECIAL / 2 - padding
    x_max = sep_x + TOKEN_BOX_W_SPECIAL / 2 + padding
    
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    
    # Create figure with proper aspect ratio
    fig_scale = 10
    fig, ax = plt.subplots(1, 1, figsize=(canvas_width * fig_scale, canvas_height * fig_scale))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')
    ax.set_aspect('equal')

    all_token_positions = []
    cp_token_positions = []

    for coup_idx, (line1, line2) in enumerate(couplets):
        # Reverse the y order: first couplet (coup_idx=0) at top, last couplet at bottom
        y_tokens = y_start + (len(couplets) - 1 - coup_idx) * line_spacing
        
        line_positions = []
        
        # Draw [CLS] attached to the left of first couplet (pushed further left)
        if coup_idx == 0:
            draw_token_box(ax, cls_x, y_tokens, '[CLS]', COLORS['special_token'], 
                          width=TOKEN_BOX_W_SPECIAL, height=TOKEN_BOX_H, fontsize=SPECIAL_TOKEN_FONT_SIZE, is_chinese=False)
            line_positions.append(cls_x)
        
        # Column 0: [CPn] token (shifted left to create equal spacing with first char)
        cp_token = f'[CP{coup_idx+1}]'
        cp_x = column_x[0] - special_offset
        draw_token_box(ax, cp_x, y_tokens, cp_token, COLORS['special_token'],
                      width=TOKEN_BOX_W_SPECIAL, height=TOKEN_BOX_H, fontsize=SPECIAL_TOKEN_FONT_SIZE, is_chinese=False)
        line_positions.append(cp_x)
        cp_token_positions.append((cp_x, y_tokens + TOKEN_BOX_H/2))
        
        # Columns 1-5: first 5 characters
        for i, ch in enumerate(line1):
            draw_token_box(ax, column_x[1 + i], y_tokens, ch, COLORS['text_token'],
                          width=TOKEN_BOX_W_ZH, height=TOKEN_BOX_H, fontsize=TOKEN_FONT_SIZE, is_chinese=True)
            line_positions.append(column_x[1 + i])
        
        # Column 6: comma (with text offset)
        draw_token_box(ax, column_x[6], y_tokens, '，', COLORS['text_token'],
                      width=TOKEN_BOX_W_ZH, height=TOKEN_BOX_H, fontsize=TOKEN_FONT_SIZE, is_chinese=True,
                      text_offset_x=PUNCT_TEXT_OFFSET)
        line_positions.append(column_x[6])
        
        # Columns 7-11: second 5 characters
        for i, ch in enumerate(line2):
            draw_token_box(ax, column_x[7 + i], y_tokens, ch, COLORS['text_token'],
                          width=TOKEN_BOX_W_ZH, height=TOKEN_BOX_H, fontsize=TOKEN_FONT_SIZE, is_chinese=True)
            line_positions.append(column_x[7 + i])
        
        # Column 12: period (with text offset)
        draw_token_box(ax, column_x[12], y_tokens, '。', COLORS['text_token'],
                      width=TOKEN_BOX_W_ZH, height=TOKEN_BOX_H, fontsize=TOKEN_FONT_SIZE, is_chinese=True,
                      text_offset_x=PUNCT_TEXT_OFFSET)
        line_positions.append(column_x[12])
        
        # Draw [SEP] attached to the right of last couplet (pushed further right)
        if coup_idx == 3:
            draw_token_box(ax, sep_x, y_tokens, '[SEP]', COLORS['special_token'],
                          width=TOKEN_BOX_W_SPECIAL, height=TOKEN_BOX_H, fontsize=SPECIAL_TOKEN_FONT_SIZE, is_chinese=False)
            line_positions.append(sep_x)
        
        all_token_positions.append((line_positions, y_tokens))

    # BERT encoder
    draw_wide_box(ax, 0.5, y_bert, 'BERT Encoder', COLORS['bert'], width=0.75, height=BERT_H)

    # Arrows from top row to BERT (including [CLS])
    top_row_positions, _ = all_token_positions[0]
    for x in top_row_positions:
        draw_arrow(ax, (x, tokens_top + ARROW_PAD), (x, bert_bottom - ARROW_PAD))
    # Arrow for [SEP] (which is on the bottom row but needs arrow from top row height)
    draw_arrow(ax, (sep_x, tokens_top + ARROW_PAD), (sep_x, bert_bottom - ARROW_PAD))

    # 4 CP embeddings
    embed_spacing = 0.15
    embed_start_x = 0.5 - 1.5 * embed_spacing

    for i in range(4):
        x_emb = embed_start_x + i * embed_spacing
        # Add CP1, CP2, CP3, CP4 labels on top of each embedding circle
        label = f'CP{i+1}'
        draw_embedding_circle(ax, x_emb, y_embed, label, radius=EMBED_RADIUS, label_position='middle')
        draw_arrow(ax, (x_emb, bert_top + ARROW_PAD), (x_emb, embed_bottom - ARROW_PAD))

    # Add "CP embeddings" annotation to the right of the last embedding circle
    last_embed_x = embed_start_x + 3 * embed_spacing
    ax.text(last_embed_x + EMBED_RADIUS + 0.02, y_embed, 'CP embeddings', ha='left', va='center',
            fontsize=LABEL_FONT_SIZE, fontfamily=FONT_EN, style='italic', color='#4a5568')

    # Single classifier (wide enough to span all 4 embeddings)
    clf_width = 3 * embed_spacing + 0.12  # spans from first to last embedding
    draw_wide_box(ax, 0.5, y_classifier, 'Linear Classifier', COLORS['classifier'], width=clf_width, height=CLF_H)
    
    # Arrows from embeddings into classifier
    for i in range(4):
        x_emb = embed_start_x + i * embed_spacing
        draw_arrow(ax, (x_emb, embed_top + ARROW_PAD), (x_emb, clf_bottom - ARROW_PAD))

    # 4 outputs
    for i in range(4):
        x_out = embed_start_x + i * embed_spacing
        label = f'P{i+1}'
        draw_token_box(ax, x_out, y_output, label, COLORS['output'], width=OUTPUT_W, height=OUTPUT_H, fontsize=OUTPUT_FONT_SIZE)
        draw_arrow(ax, (x_out, clf_top + ARROW_PAD), (x_out, output_bottom - ARROW_PAD))

    # Place annotation to the right of the last output
    last_x = embed_start_x + 3 * embed_spacing
    ax.text(last_x + OUTPUT_W/2 + 0.02, y_output, '4 Parallel Labels', ha='left', va='center',
            fontsize=LABEL_FONT_SIZE, fontfamily=FONT_EN, style='italic', color='#718096')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'arch_poem4_model.png', bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# 4. Poem-1 Model (Full Poem, 1 Output)
# =============================================================================

def draw_poem1_model():
    # Example poem (4 couplets)
    couplets = [
        ('感別情偏切', '離筵酒易酣'),
        ('孤舟搖霽月', '一騎入晴嵐'),
        ('戀闕心懸北', '思親夢繞南'),
        ('明朝倚樓處', '雲樹思難堪'),
    ]

    # Token spacing for this diagram
    spacing = 0.048
    line_spacing = 0.06
    padding = 0.04
    special_offset = 0.008

    # Calculate column positions for aligned tokens
    n_columns = 12  # char1-5, comma, char6-10, period
    total_token_width = (n_columns - 1) * spacing

    # Vertical positions (from bottom up)
    y_start = padding
    top_row_y = y_start + 3 * line_spacing
    tokens_top = top_row_y + TOKEN_BOX_H / 2
    
    bert_bottom = tokens_top + ARROW_GAP
    y_bert = bert_bottom + BERT_H / 2
    bert_top = y_bert + BERT_H / 2
    
    embed_bottom = bert_top + ARROW_GAP
    y_embed = embed_bottom + EMBED_RADIUS
    embed_top = y_embed + EMBED_RADIUS
    
    clf_bottom = embed_top + ARROW_GAP
    y_classifier = clf_bottom + CLF_H / 2
    clf_top = y_classifier + CLF_H / 2
    
    output_bottom = clf_top + ARROW_GAP
    y_output = output_bottom + OUTPUT_H / 2
    output_top = y_output + OUTPUT_H / 2

    # Calculate canvas bounds
    y_min = 0
    y_max = output_top + padding
    
    # Horizontal: include CLS on left, SEP on right
    start_x = 0.5 - total_token_width / 2
    column_x = [start_x + i * spacing for i in range(n_columns)]
    cls_x = column_x[0] - spacing - special_offset
    sep_x = column_x[11] + spacing + special_offset
    
    x_min = cls_x - TOKEN_BOX_W_SPECIAL / 2 - padding
    x_max = sep_x + TOKEN_BOX_W_SPECIAL / 2 + padding
    
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    
    # Create figure with proper aspect ratio
    fig_scale = 10
    fig, ax = plt.subplots(1, 1, figsize=(canvas_width * fig_scale, canvas_height * fig_scale))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')
    ax.set_aspect('equal')

    all_token_positions = []

    for coup_idx, (line1, line2) in enumerate(couplets):
        # Reverse the y order: first couplet (coup_idx=0) at top, last couplet at bottom
        y_tokens = y_start + (len(couplets) - 1 - coup_idx) * line_spacing
        
        line_positions = []
        
        # Draw [CLS] attached to the left of first couplet (pushed further left)
        if coup_idx == 0:
            draw_token_box(ax, cls_x, y_tokens, '[CLS]', COLORS['special_token'],
                          width=TOKEN_BOX_W_SPECIAL, height=TOKEN_BOX_H, fontsize=SPECIAL_TOKEN_FONT_SIZE, is_chinese=False)
            line_positions.append(cls_x)
        
        # Columns 0-4: first 5 characters
        for i, ch in enumerate(line1):
            draw_token_box(ax, column_x[i], y_tokens, ch, COLORS['text_token'],
                          width=TOKEN_BOX_W_ZH, height=TOKEN_BOX_H, fontsize=TOKEN_FONT_SIZE, is_chinese=True)
            line_positions.append(column_x[i])
        
        # Column 5: comma (with text offset)
        draw_token_box(ax, column_x[5], y_tokens, '，', COLORS['text_token'],
                      width=TOKEN_BOX_W_ZH, height=TOKEN_BOX_H, fontsize=TOKEN_FONT_SIZE, is_chinese=True,
                      text_offset_x=PUNCT_TEXT_OFFSET)
        line_positions.append(column_x[5])
        
        # Columns 6-10: second 5 characters
        for i, ch in enumerate(line2):
            draw_token_box(ax, column_x[6 + i], y_tokens, ch, COLORS['text_token'],
                          width=TOKEN_BOX_W_ZH, height=TOKEN_BOX_H, fontsize=TOKEN_FONT_SIZE, is_chinese=True)
            line_positions.append(column_x[6 + i])
        
        # Column 11: period (with text offset)
        draw_token_box(ax, column_x[11], y_tokens, '。', COLORS['text_token'],
                      width=TOKEN_BOX_W_ZH, height=TOKEN_BOX_H, fontsize=TOKEN_FONT_SIZE, is_chinese=True,
                      text_offset_x=PUNCT_TEXT_OFFSET)
        line_positions.append(column_x[11])
        
        # Draw [SEP] attached to the right of last couplet (pushed further right)
        if coup_idx == 3:
            draw_token_box(ax, sep_x, y_tokens, '[SEP]', COLORS['special_token'],
                          width=TOKEN_BOX_W_SPECIAL, height=TOKEN_BOX_H, fontsize=SPECIAL_TOKEN_FONT_SIZE, is_chinese=False)
            line_positions.append(sep_x)
        
        all_token_positions.append((line_positions, y_tokens))

    # BERT encoder
    draw_wide_box(ax, 0.5, y_bert, 'BERT Encoder', COLORS['bert'], width=0.65, height=BERT_H)

    # Arrows from top row to BERT (including [CLS])
    top_row_positions, _ = all_token_positions[0]
    for x in top_row_positions:
        draw_arrow(ax, (x, tokens_top + ARROW_PAD), (x, bert_bottom - ARROW_PAD))
    # Arrow for [SEP] (which is on the bottom row but needs arrow from top row height)
    draw_arrow(ax, (sep_x, tokens_top + ARROW_PAD), (sep_x, bert_bottom - ARROW_PAD))

    # Single [CLS] embedding
    draw_embedding_circle(ax, 0.5, y_embed, '[CLS] embedding', radius=EMBED_RADIUS)
    draw_arrow(ax, (0.5, bert_top + ARROW_PAD), (0.5, embed_bottom - ARROW_PAD))

    # Single classifier
    draw_wide_box(ax, 0.5, y_classifier, 'Linear Classifier', COLORS['classifier'], width=0.32, height=CLF_H)
    draw_arrow(ax, (0.5, embed_top + ARROW_PAD), (0.5, clf_bottom - ARROW_PAD))

    # Single output
    draw_token_box(ax, 0.5, y_output, 'P / not P', COLORS['output'], width=OUTPUT_W, height=OUTPUT_H, fontsize=OUTPUT_FONT_SIZE)
    draw_arrow(ax, (0.5, clf_top + ARROW_PAD), (0.5, output_bottom - ARROW_PAD))

    ax.text(0.5 + OUTPUT_W/2 + 0.02, y_output, 'Parallel / Not Parallel', ha='left', va='center',
            fontsize=LABEL_FONT_SIZE, fontfamily=FONT_EN, style='italic', color='#718096')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'arch_poem1_model.png', bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Generating model architecture diagrams...")
    draw_char_model()
    print("  - arch_char_model.png")
    draw_coup_model()
    print("  - arch_coup_model.png")
    draw_poem4_model()
    print("  - arch_poem4_model.png")
    draw_poem1_model()
    print("  - arch_poem1_model.png")
    print("Done!")

