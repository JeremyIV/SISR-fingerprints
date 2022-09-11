import database.api as db

query = (
    "select g.name name, i.psnr psnr, i.lpips lpips"
    " from image_patch i"
    " inner join generator g on i.generator_id = g.id"
    " where (i.crop_lower - i.crop_upper) = 224"
    " and i.psnr is not null"
    " and i.lpips is not null"
)

results = db.read_sql_query(query)

grouped_results = results.groupby("name")
means = grouped_results.mean()
stds = grouped_results.std()


def fmt2(num):
    return f"{num:.02f}"


def fmt3(num):
    return f"{num:.03f}"


table = ""
for name in sorted(means.index):
    psnr_mean = fmt2(means.psnr[name])
    psnr_std = fmt2(stds.psnr[name])
    lpips_mean = fmt3(means.lpips[name])
    lpips_std = fmt3(stds.lpips[name])

    name = name.replace("VGG_GAN", "VGG+Adv.")
    name = name.replace("ResNet_GAN", "ResNet+Adv.")
    name = name.replace("quarter_", "quarter ")
    name = name.replace("_", " ")
    table += f"{name} & ${psnr_mean} (\\pm {psnr_std})$ & ${lpips_mean} (\\pm {lpips_std})$ \\\\\n"

with open("analysis/values/sisr_psnr_lpips_table.txt", "w") as f:
    f.write(table)
