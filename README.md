## Create mask param behaviour

### I 
`choose_largest_mask:  True`: The create_mask_server node will automatically choosoe the largest mask among the masks it got from SAM. 'choose_largest_mask' has no effect
### II 
`choose_largest_mask: False` && `inspect_masks: False`: The create_mask_server node will loop through all masks and return the largest one with five contours. If no mask has five contours the largest one is returned.
### III 
`choose_largest_mask: False` && `inspect_masks: True`: Same as II + ALL found masks are displayed (with cv2.waitKey(0)). The largest one is returned.



