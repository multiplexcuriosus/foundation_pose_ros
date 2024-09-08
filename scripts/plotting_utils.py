import rospy
import numpy as np
import matplotlib.pyplot as plt


def get_seg_points(image):
    object_points = []
    background_points = []
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('on')
    plt.title("Select object points with left click and background points with right click.\nPress ENTER to finish.", fontsize=12)
    plt.connect(
        'button_press_event',
        lambda event: onclickbutton(event, object_points=object_points, background_points=background_points)
    )
    plt.connect('key_press_event', lambda event: plt.close() if event.key == 'enter' else None)
    plt.show()
    plt.disconnect('button_press_event')
    plt.disconnect('key_press_event')
    plt.close()
    object_points = np.array(object_points).reshape(-1, 2)
    background_points = np.array(background_points).reshape(-1, 2)
    points = np.concatenate([object_points, background_points], axis=0)
    labels = np.concatenate([np.ones(len(object_points)), np.zeros(len(background_points))], axis=0)
    return points, labels


def onclickbutton(event, object_points, background_points):
    if event.button == 1:
        object_points.append((event.xdata, event.ydata))
        #rospy.loginfo(f"[CreateMaskNode]: Left click at: {event.xdata}, {event.ydata}. Object point added.")
        event.inaxes.scatter(
            event.xdata, event.ydata, color='green', marker='*', s=200, edgecolor='white', linewidth=1.25
        )
    elif event.button == 3:
        background_points.append((event.xdata, event.ydata))
        #rospy.loginfo(f"[CreateMaskNode]: Right Click at: {event.xdata}, {event.ydata}. Background point added.")
        event.inaxes.scatter(
            event.xdata, event.ydata, color='red', marker='*', s=200, edgecolor='white', linewidth=1.25
        )
    else:
        return
    #rospy.loginfo(f"[CreateMaskNode]: Total object points: {len(object_points)}, Total background points: {len(background_points)}\n----\n")
    plt.draw()


def show_masks(image, points, labels, masks, scores, axs):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        ax = axs[i]
        ax.imshow(image)
        draw_mask(mask, ax)
        # draw_points(points, labels, ax)
        ax.set_title(f"Mask {i+1}, Score: {score:.3f}")
        # ax.axis('off')


def draw_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 144 / 255, 40 / 255, 1.0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=0.8)


def draw_points(points, labels, ax, marker_size=150):
    pos_points = points[labels == 1]
    neg_points = points[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25
    )


def draw_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def select_mask(fig, axs):
    ax_id = {"index": -1}
    fig.canvas.mpl_connect('button_press_event', lambda event: onclickaxis(event, fig, axs.flatten().tolist(), ax_id))
    plt.show()
    if ax_id["index"] >= 0:
        fig.canvas.mpl_disconnect('button_press_event')
        return ax_id["index"]
    return None


def onclickaxis(event, fig, ax_list, ax_index):
    clicked_ax = event.inaxes
    if clicked_ax is None:
        ax_index["index"] = -1
        return
    ax_index["index"] = ax_list.index(clicked_ax)
    #rospy.loginfo(f"[CreateMaskNode]: Selected mask {ax_index['index'] + 1}. You can close the window now.")
    plt.close(fig)
