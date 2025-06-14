# ACM Certificate DNS Validation Records

Certificate ARN: `arn:aws:acm:us-east-1:124355672559:certificate/183ffae7-82d7-4259-a773-f52bb05c46d8`

## DNS Records to Add

Please add the following CNAME records to your DNS provider (Squarespace):

### For clarity.novamindnyc.com:
- **Name**: `_b40e5e1eca8c6fe5aab4043fea22fa97.clarity`
- **Type**: CNAME
- **Value**: `_1c8ca645dd856273474dd343f4002a41.xlfgrmvvlj.acm-validations.aws.`
- **TTL**: 300 seconds (or default)

### For novamindnyc.com (apex domain):
- **Name**: `_dafb0dc4788603bffe67544f6dc3f34e`
- **Type**: CNAME
- **Value**: `_a4475b24a4a74cfd0be51afe4aac43c2.xlfgrmvvlj.acm-validations.aws.`
- **TTL**: 300 seconds (or default)

## Important Notes

1. The trailing dots (.) in the values are important - include them if your DNS provider supports them
2. For the apex domain (novamindnyc.com), you may need to use the @ symbol or leave the name field blank depending on your DNS provider
3. DNS propagation can take up to 30 minutes
4. The certificate will automatically validate once AWS can resolve these records

## Verification

After adding the records, you can verify they're working:

```bash
# Check clarity subdomain validation
dig +short _b40e5e1eca8c6fe5aab4043fea22fa97.clarity.novamindnyc.com CNAME

# Check apex domain validation  
dig +short _dafb0dc4788603bffe67544f6dc3f34e.novamindnyc.com CNAME
```

## Next Steps

Once these DNS records are added and propagated, the certificate will automatically validate and move to "Issued" status. This typically takes 5-30 minutes.