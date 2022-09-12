import math
import pygame as pg
import matplotlib.pyplot as plt


def plot(mean_scores, sum_rewards, file_name):
    # see: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html

    # 1
    _, ax1 = plt.subplots()
    plt.title("Training...")

    color = "tab:red"
    ax1.set_xlabel("Games")
    ax1.set_ylabel("Mean scores", color=color)
    ax1.plot(mean_scores, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    value = round(mean_scores[-1], 1)
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(value), color=color)

    # 2
    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel(
        "Sum rewards",
        color=color,
    )
    ax2.plot(sum_rewards, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    value = sum_rewards[-1]
    plt.text(len(sum_rewards) - 1, sum_rewards[-1], str(value), color=color)

    # plot/saving
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.025)
    # plt.show()


def message(
    screen, msg, font_size, color, position, anchor="topleft", rotation=0
) -> None:
    """
    Displays a message on screen.

    Args:
        msg (string, required): text to be displayed
        font_size (int, required): font size
        color (tuple, required): text RGB color code
        position (tuple, required): text position on screen
        anchor (string, optional, default="topleft"): anchor position on screen
        rotation (float, optional, default=0): text rotation    

    Returns:
        None
        
    Raises:
        AttributeError: exception raised when anchor is invalid
    """
    font = pg.font.SysFont("Calibri", font_size)
    text = font.render(msg, True, color)
    text = pg.transform.rotate(text, rotation)
    
    if anchor == "topleft":
        text_rect = text.get_rect(topleft=(position))
    elif anchor == "center":
        text_rect = text.get_rect(center=(position))
    else:
        raise AttributeError("Invalid anchor!")
        
    screen.blit(text, text_rect)


def progress_bar(screen, x, y, w_bg, h_bg, w_fg, h_fg, bg_color, fg_color) -> None:
    """"
    Draws a progress bar on screen.
    
    Args:
        x (int, required): bars x coordinate
        y (int, required): bars y coordinate
        w_bg (float, required): background bar width
        h_bg (float, required): background bar height
        w_fg (float, required): foreground bar width
        h_fg (float, required): foreground bar height
        bg_color (tuple, required): background RGB color code
        fg_color (tuple, required): foreground RGB color code
        
    Returns:
        None
        
    Raises:
        None
    """
    pg.draw.rect(screen, bg_color, (x, y, w_bg, h_bg))
    pg.draw.rect(screen, fg_color, (x, y, w_fg, h_fg), width=2)


def distance(p1, p2) -> float:
    """Returns the distance between two points p1, p2.
    
    Agrs:
        p1 (tuple, required): coordinates of point p1
        p2 (tuple, required): coordinates of point p2
        
    Returns:
        resulting distance
        
    Raises:
        None
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
