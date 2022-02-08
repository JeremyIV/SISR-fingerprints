-- create_views.sql
-- analysis, joins together all these tables. same cardinality as predictions.
create view analysis as
    select 
        p.actual as actual,
        p.predicted as predicted,
        p.feature as feature,
        p.class_probabilities as class_probabilities,
        i.image_path as image_path,
        i.patch_hash as patch_hash,
        i.crop_upper as crop_upper,
        i.crop_left as crop_left,
        i.crop_lower as crop_lower,
        i.crop_right as crop_right,
        i.acutance as acutance,
        i.psnr as psnr,
        i.lpips as lpips,
        g.name as generator_name,
        c.name as classifier_name,
        c.path as classifier_path,
        c.type as classifier_type,
        c.opt as classifier_opt,
        d.type as dataset_type
        d.name as dataset_name
        d.ordered_labels as ordered_labels
        d.opt as dataset_opt
    from prediction p
        inner join image_patch i on i.id = p.image_patch_id
        inner join generator g on g.id = i.generator_id
        inner join classifier c on c.id = p.classifier_id
        inner join dataset d on d.id = p.dataset_id;

create view sisr_analysis as
    select 
        p.actual as actual,
        p.predicted as predicted,
        p.feature as feature,
        p.class_probabilities as class_probabilities,
        i.image_path as image_path,
        i.patch_hash as patch_hash,
        i.crop_upper as crop_upper,
        i.crop_left as crop_left,
        i.crop_lower as crop_lower,
        i.crop_right as crop_right,
        i.acutance as acutance,
        i.psnr as psnr,
        i.lpips as lpips,
        g.name as generator_name,
        sg.architecture as architecture,
        sg.dataset as dataset,
        sg.scale as scale,
        sg.loss as loss,
        sg.seed as seed,
        c.name as classifier_name,
        c.path as classifier_path,
        c.type as classifier_type,
        c.opt as classifier_opt,
        d.type as dataset_type,
        d.name as dataset_name,
        d.ordered_labels as ordered_labels,
        d.opt as dataset_opt,
        sd.label_param as label_param,
        sd.reserved_param as reserved_param,
        sd.reserved_param_value as reserved_param_value,
        sd.include_pretrained as include_pretrained,
        sd.include_custom_trained as include_custom_trained,
    from prediction p
        inner join image_patch i on i.id = p.image_patch_id
        inner join generator g on g.id = i.generator_id
        inner join classifier c on c.id = p.classifier_id
        inner join dataset d on d.id = p.dataset_id
        inner join SISR_generator sg on sg.generator_id = g.id
        inner join SISR_dataset sd on sd.dataset_id = d.id;